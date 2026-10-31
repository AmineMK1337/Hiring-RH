# Semantic (embedding-based) scoring of candidates against a job offer
"""Requirement-level semantic ranking.

For each job requirement, find the candidate's best-matching chunk.
Calibrate that similarity to 0..1, then take a weighted mean
(required > preferred). Every requirement keeps its evidence chunk,
so the score is explainable and the recruiter can check it.

Ingestion (upsert) and ranking are separate: chunks are embedded once
when a resume arrives; ranking a job is then a matrix product.
"""
from collections.abc import Iterable
from dataclasses import asdict

import numpy as np

from core.llm.embeddings import Embedder
from services.screener.config import SemanticConfig
from services.screener.models import (
    CandidateDocument,
    JobOffer,
    RequirementEvidence,
    SemanticMatch,
    SemanticRanking,
)
from services.screener.scoring.chunking import Chunk, chunk_candidate


class SemanticIndex:
    def __init__(self, embedder: Embedder, config: SemanticConfig | None = None):
        self._embedder = embedder
        self.config = config or SemanticConfig()
        self._chunks: dict[str, list[Chunk]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._flags: dict[str, bool] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def upsert(self, docs: Iterable[CandidateDocument]) -> None:
        """Index or re-index candidates. Re-indexing replaces old chunks."""
        per_doc = [
            (
                doc,
                chunk_candidate(
                    doc,
                    max_chars=self.config.max_chunk_chars,
                    max_chunks=self.config.max_chunks_per_candidate,
                ),
            )
            for doc in docs
        ]
        all_texts = [c.text for _, chunks in per_doc for c in chunks]
        vectors = self._embedder.embed_passages(all_texts)  # one batch call
        offset = 0
        for doc, chunks in per_doc:
            n = len(chunks)
            self._chunks[doc.candidate_id] = chunks
            self._vectors[doc.candidate_id] = vectors[offset : offset + n]
            self._flags[doc.candidate_id] = doc.injection_flag
            offset += n

    def remove(self, candidate_id: str) -> None:
        self._chunks.pop(candidate_id, None)
        self._vectors.pop(candidate_id, None)
        self._flags.pop(candidate_id, None)

    def chunk_text(self, chunk_id: str) -> str | None:
        """Evidence text for the recruiter UI. Never feed it to an LLM unfenced."""
        candidate_id = chunk_id.rsplit(":", 1)[0]
        for chunk in self._chunks.get(candidate_id, []):
            if chunk.chunk_id == chunk_id:
                return chunk.text
        return None

    def rank(self, job: JobOffer, top_k: int | None = None) -> SemanticRanking:
        req_vecs = self._embedder.embed_queries([r.text for r in job.requirements])
        weights = np.array(
            [self.config.weights[r.kind] for r in job.requirements], dtype=np.float32
        )

        scored = [
            self._score_candidate(cid, job, req_vecs, weights) for cid in self._chunks
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))  # deterministic ties
        if top_k is not None:
            scored = scored[:top_k]

        matches = [
            SemanticMatch(
                candidate_id=cid,
                rank=position,
                score=round(score, 4),
                requirements_met=met,
                requirements_total=len(job.requirements),
                injection_flag=self._flags[cid],
                evidence=evidence,
            )
            for position, (score, cid, met, evidence) in enumerate(scored, start=1)
        ]
        return SemanticRanking(
            job_id=job.job_id,
            model_id=self._embedder.model_id,
            config=asdict(self.config),
            matches=matches,
        )

    def _score_candidate(
        self, cid: str, job: JobOffer, req_vecs: np.ndarray, weights: np.ndarray
    ) -> tuple[float, str, int, list[RequirementEvidence]]:
        chunks = self._chunks[cid]
        n_req = len(job.requirements)

        if chunks:
            sims = req_vecs @ self._vectors[cid].T          # (requirements, chunks)
            best_pos = sims.argmax(axis=1)
            best_raw = sims[np.arange(n_req), best_pos]
            calibrated = self.config.calibrate(best_raw)
            best_ids = [chunks[p].chunk_id for p in best_pos]
        else:
            best_raw = np.zeros(n_req, dtype=np.float32)
            calibrated = np.zeros(n_req, dtype=np.float32)
            best_ids = [None] * n_req

        score = float((weights * calibrated).sum() / weights.sum())
        met = int((calibrated >= self.config.met_threshold).sum())
        evidence = [
            RequirementEvidence(
                requirement=req.text,
                kind=req.kind,
                score=round(float(calibrated[i]), 4),
                raw_similarity=round(float(best_raw[i]), 4),
                best_chunk_id=best_ids[i],
            )
            for i, req in enumerate(job.requirements)
        ]
        return score, cid, met, evidence
