import numpy as np
import pytest

from core.llm.embeddings import HashingEmbedder
from services.screener.config import SemanticConfig
from services.screener.models import (
    CandidateDocument,
    JobOffer,
    Requirement,
    ResumeSection,
)
from services.screener.scoring.semantic import SemanticIndex


def make_index() -> SemanticIndex:
    return SemanticIndex(HashingEmbedder(), SemanticConfig(floor=0.0, ceiling=1.0))


def doc(cid: str, *lines: str, flag: bool = False) -> CandidateDocument:
    return CandidateDocument(
        candidate_id=cid,
        sections=[ResumeSection(kind="experience", text="\n".join(lines))],
        injection_flag=flag,
    )


JOB = JobOffer(
    job_id="job-1",
    title="Backend",
    requirements=[
        Requirement(text="Build REST APIs in Python", kind="required"),
        Requirement(text="Write automated tests", kind="preferred"),
    ],
)
from services.screener.scoring.semantic import SemanticIndex


def test_relevant_candidate_ranks_first():
    index = make_index()
    index.upsert([
        doc("bad", "Managed social media marketing campaigns."),
        doc("good", "Built REST APIs in Python with Django.", "Wrote automated tests with pytest."),
    ])
    ranking = index.rank(JOB)
    assert [m.candidate_id for m in ranking.matches] == ["good", "bad"]
    assert ranking.matches[0].rank == 1


def test_repeating_lines_adds_nothing():
    index = make_index()
    line = "Built REST APIs in Python."
    index.upsert([doc("once", line), doc("stuffed", *([line] * 50))])
    scores = {m.candidate_id: m.score for m in index.rank(JOB).matches}
    assert scores["once"] == scores["stuffed"]


def test_reindex_replaces_old_chunks():
    index = make_index()
    index.upsert([doc("c1", "Organized company events.")])
    before = index.rank(JOB).matches[0].score
    index.upsert([doc("c1", "Built REST APIs in Python.")])
    after = index.rank(JOB).matches[0].score
    assert len(index) == 1
    assert after > before


def test_empty_resume_scores_zero():
    index = make_index()
    index.upsert([CandidateDocument(candidate_id="empty", sections=[])])
    match = index.rank(JOB).matches[0]
    assert match.score == 0.0
    assert all(ev.best_chunk_id is None for ev in match.evidence)


def test_ties_are_deterministic():
    index = make_index()
    index.upsert([doc("b", "Built REST APIs."), doc("a", "Built REST APIs.")])
    assert [m.candidate_id for m in index.rank(JOB).matches] == ["a", "b"]


def test_top_k_and_evidence_lookup():
    index = make_index()
    index.upsert([doc(f"c{i}", "Built REST APIs in Python.") for i in range(5)])
    ranking = index.rank(JOB, top_k=2)
    assert len(ranking.matches) == 2
    chunk_id = ranking.matches[0].evidence[0].best_chunk_id
    assert index.chunk_text(chunk_id) == "Built REST APIs in Python."


def test_injection_flag_is_carried_not_filtered():
    index = make_index()
    index.upsert([doc("flagged", "Built REST APIs in Python.", flag=True)])
    match = index.rank(JOB).matches[0]
    assert match.injection_flag is True
    assert match.score > 0


def test_calibration_is_clipped():
    config = SemanticConfig(floor=0.5, ceiling=0.9)
    out = config.calibrate(np.array([0.1, 0.5, 0.7, 0.9, 1.0]))
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        SemanticConfig(floor=0.9, ceiling=0.8)


def test_ranking_records_model_and_config():
    index = make_index()
    index.upsert([doc("c1", "Built REST APIs.")])
    ranking = index.rank(JOB)
    assert ranking.model_id.startswith("hashing-bow")
    assert ranking.config["floor"] == 0.0
