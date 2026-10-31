"""Rank the sample candidates for one job and print the evidence.

    python scripts/demo_screener_semantic.py --job job-backend-python
    python scripts/demo_screener_semantic.py --job job-data-analyst --fake   # no model download
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.llm.embeddings import HashingEmbedder, SentenceTransformerEmbedder  # noqa: E402
from services.screener.config import EMBEDDING_MODEL, SemanticConfig  # noqa: E402
from services.screener.models import CandidateDocument, JobOffer  # noqa: E402
from services.screener.scoring.semantic import SemanticIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="job-backend-python")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--fake", action="store_true", help="bag-of-words embedder, no download")
    parser.add_argument("--evidence", type=int, default=3, help="show evidence for top N")
    args = parser.parse_args()

    jobs_raw = json.loads((ROOT / "data/job_offers/sample_job_offers.json").read_text())
    cands_raw = json.loads((ROOT / "data/resumes/sample_candidates_sanitized.json").read_text())
    jobs = {j["job_id"]: JobOffer(**j) for j in jobs_raw}
    candidates = [CandidateDocument(**c) for c in cands_raw]
    job = jobs[args.job]

    if args.fake:
        index = SemanticIndex(HashingEmbedder(), SemanticConfig(floor=0.0, ceiling=0.6))
    else:
        index = SemanticIndex(SentenceTransformerEmbedder(EMBEDDING_MODEL))
    index.upsert(candidates)
    ranking = index.rank(job, top_k=args.top)

    print(f"\n{job.title}  (model: {ranking.model_id})\n")
    print(f"{'rank':<5}{'candidate':<12}{'score':>7}  met")
    for m in ranking.matches:
        flag = "  [injection flag]" if m.injection_flag else ""
        print(f"{m.rank:<5}{m.candidate_id:<12}{m.score:>7.3f}  "
              f"{m.requirements_met}/{m.requirements_total}{flag}")

    for m in ranking.matches[: args.evidence]:
        print(f"\n{m.candidate_id}")
        for ev in m.evidence:
            text = index.chunk_text(ev.best_chunk_id) if ev.best_chunk_id else "-"
            print(f"  {ev.score:.2f} (cos {ev.raw_similarity:.3f})  {ev.requirement}")
            print(f"        <- {text}")


if __name__ == "__main__":
    main()
