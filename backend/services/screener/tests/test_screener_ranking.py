from datetime import date

import pytest

from core.llm.embeddings import HashingEmbedder
from services.screener.config import RankingWeights, SemanticConfig
from services.screener.models import (
    CandidateDocument,
    JobOffer,
    Requirement,
    ResumeSection,
    SemanticMatch,
    SemanticRanking,
)
from services.screener.scoring.ranking_engine import GraphResult, NullGraphScorer, rank_combined
from services.screener.scoring.semantic import SemanticIndex

AS_OF = date(2026, 11, 1)

JOB_3Y = JobOffer(
    job_id="job-3y",
    title="Backend Python Developer",
    requirements=[
        Requirement(text="At least 3 years of experience building REST APIs in Python"),
        Requirement(text="Containerize applications with Docker", kind="preferred"),
    ],
)
JOB_NO_REQ = JobOffer(
    job_id="job-open",
    title="Backend Python Developer",
    requirements=[Requirement(text="Build REST APIs in Python")],
)


class FixedSemantic:
    """Stub ranker returning fixed semantic scores, to isolate the other criteria."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def rank(self, job, top_k=None):
        return SemanticRanking(
            job_id=job.job_id, model_id="fixed", config={},
            matches=[
                SemanticMatch(candidate_id=cid, rank=0, score=s, requirements_met=0,
                              requirements_total=len(job.requirements), injection_flag=False,
                              evidence=[])
                for cid, s in self.scores.items()
            ],
        )


class FixedGraph:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, job, candidate_ids):
        return {cid: GraphResult(self.scores[cid]) for cid in candidate_ids}


def cand(cid: str, experience: str, skills: str = "") -> CandidateDocument:
    sections = [ResumeSection(kind="experience", text=experience)]
    if skills:
        sections.append(ResumeSection(kind="skills", text=skills))
    return CandidateDocument(candidate_id=cid, sections=sections)


def rank(job, docs, semantic_scores, **kwargs):
    return rank_combined(job, docs, FixedSemantic(semantic_scores), as_of=AS_OF, **kwargs)


BACKEND_3Y = "Backend developer | Jan 2021 - Dec 2023\n- Built REST APIs in Python"
BACKEND_1Y = "Backend developer | Jan 2023 - Dec 2023\n- Built REST APIs in Python"


# ---------------------------------------------------------------- N/A handling

def test_no_requirement_ranks_by_remaining_criteria_only():
    docs = [cand("a", BACKEND_1Y), cand("b", BACKEND_3Y)]
    ranking = rank(JOB_NO_REQ, docs, {"a": 0.8, "b": 0.6})
    assert ranking.required_years is None
    assert [c.candidate_id for c in ranking.candidates] == ["a", "b"]
    for c in ranking.candidates:
        assert c.scores.experience is None
        assert c.applied_criteria == ["semantic"]
        assert c.weights_used == {"semantic": 1.0}
        assert c.total_score == c.scores.semantic


def test_requirement_applies_experience_and_renormalizes_weights():
    ranking = rank(JOB_3Y, [cand("a", BACKEND_3Y)], {"a": 0.6})
    c = ranking.candidates[0]
    assert ranking.required_years == 3
    assert c.applied_criteria == ["semantic", "experience"]
    assert c.weights_used == pytest.approx({"semantic": 0.5 / 0.7, "experience": 0.2 / 0.7}, abs=1e-4)
    assert c.total_score == pytest.approx((0.5 * 0.6 + 0.2 * 1.0) / 0.7, abs=1e-4)
    assert ranking.weights_config == {"semantic": 0.5, "experience": 0.2, "graph": 0.3}


def test_null_graph_scorer_keeps_ranking_working():
    ranking = rank(JOB_3Y, [cand("a", BACKEND_3Y)], {"a": 0.5}, graph_scorer=NullGraphScorer())
    c = ranking.candidates[0]
    assert c.scores.graph is None and "graph" not in c.applied_criteria
    assert ranking.graph_scorer == "NullGraphScorer"


def test_graph_scorer_plugs_in_without_changes():
    ranking = rank(JOB_3Y, [cand("a", BACKEND_3Y)], {"a": 0.5}, graph_scorer=FixedGraph({"a": 0.9}))
    c = ranking.candidates[0]
    assert c.applied_criteria == ["semantic", "experience", "graph"]
    assert sum(c.weights_used.values()) == pytest.approx(1.0)
    assert c.scores.graph == 0.9


def test_weights_are_validated():
    with pytest.raises(ValueError):
        RankingWeights(semantic=-1.0)
    with pytest.raises(ValueError):
        RankingWeights(semantic=0.0, experience=0.0, graph=0.0)
    with pytest.raises(ValueError):
        RankingWeights(semantic=float("nan"))


def test_zero_weight_criterion_is_not_applied():
    weights = RankingWeights(semantic=1.0, experience=0.0, graph=0.0)
    c = rank(JOB_3Y, [cand("a", BACKEND_1Y)], {"a": 0.4}, weights=weights).candidates[0]
    assert c.applied_criteria == ["semantic"] and c.total_score == 0.4


# ---------------------------------------------------------------- determinism

def test_two_runs_are_identical():
    docs = [cand("a", BACKEND_1Y), cand("b", BACKEND_3Y), cand("c", "Call center | 2010 - 2020")]
    scores = {"a": 0.7, "b": 0.6, "c": 0.2}
    assert rank(JOB_3Y, docs, scores).model_dump() == rank(JOB_3Y, docs, scores).model_dump()


def test_ties_broken_by_candidate_id():
    docs = [cand("b", BACKEND_3Y), cand("a", BACKEND_3Y)]
    ranking = rank(JOB_3Y, docs, {"a": 0.5, "b": 0.5})
    assert [c.candidate_id for c in ranking.candidates] == ["a", "b"]
    assert [c.rank for c in ranking.candidates] == [1, 2]


def test_equal_totals_broken_by_semantic_score():
    # (0.5*0.6 + 0.2*1.0) / 0.7 == (0.5*0.8 + 0.2*0.5) / 0.7
    docs = [cand("x", BACKEND_3Y), cand("y", "Backend developer | Jul 2022 - Dec 2023\n- Python")]
    ranking = rank(JOB_3Y, docs, {"x": 0.6, "y": 0.8})
    assert ranking.candidates[0].total_score == ranking.candidates[1].total_score
    assert [c.candidate_id for c in ranking.candidates] == ["y", "x"]


def test_real_semantic_index_is_deterministic():
    docs = [cand("a", BACKEND_3Y), cand("b", "Call center agent | 2015 - 2020\n- Answered calls")]
    index = SemanticIndex(HashingEmbedder(), SemanticConfig(floor=0.0, ceiling=1.0))
    index.upsert(docs)
    first = rank_combined(JOB_3Y, docs, index, as_of=AS_OF)
    second = rank_combined(JOB_3Y, docs, index, as_of=AS_OF)
    assert first.model_dump() == second.model_dump()
    assert first.candidates[0].candidate_id == "a"
    assert first.semantic_model_id.startswith("hashing-bow")


# ---------------------------------------------------------------- fairness

def test_older_unrelated_start_dates_give_the_same_score():
    young = cand("young", BACKEND_3Y + "\n\nCall center agent | 2017 - 2020\n- Answered calls")
    older = cand("older", BACKEND_3Y + "\n\nCall center agent | 1996 - 2020\n- Answered calls")
    by_id = {c.candidate_id: c for c in rank(JOB_3Y, [young, older], {"young": 0.5, "older": 0.5}).candidates}
    assert by_id["young"].total_score == by_id["older"].total_score
    assert by_id["young"].scores == by_id["older"].scores


def test_long_unrelated_experience_does_not_beat_required_relevant_years():
    veteran = cand("veteran", "Call center agent | 2004 - 2024\n- Answered customer calls",
                   skills="Python, REST APIs, Docker")
    junior = cand("junior", BACKEND_3Y)
    ranking = rank(JOB_3Y, [veteran, junior], {"veteran": 0.5, "junior": 0.5})
    by_id = {c.candidate_id: c for c in ranking.candidates}
    assert by_id["veteran"].experience.total_years == 21
    assert by_id["veteran"].experience.relevant_years == 0
    assert ranking.candidates[0].candidate_id == "junior"


def test_additional_skills_never_change_the_score():
    docs = [
        cand("plain", BACKEND_3Y),
        cand("extra", BACKEND_3Y, skills="MongoDB, React, Tableau"),
    ]
    index = SemanticIndex(HashingEmbedder(), SemanticConfig(floor=0.0, ceiling=1.0))
    index.upsert(docs)
    by_id = {c.candidate_id: c for c in rank_combined(JOB_3Y, docs, index, as_of=AS_OF).candidates}
    assert by_id["extra"].additional_skills == ["mongodb", "react", "tableau"]
    assert by_id["plain"].additional_skills == []
    assert by_id["extra"].total_score == by_id["plain"].total_score
    assert by_id["extra"].scores == by_id["plain"].scores


# ---------------------------------------------------------------- outputs

def test_matched_and_missing_skills():
    c = rank(JOB_3Y, [cand("a", BACKEND_3Y)], {"a": 0.5}).candidates[0]
    assert c.matched_skills == ["python", "rest_api"]
    assert c.missing_skills == ["docker"]


def test_outputs_contain_no_personal_data_fields():
    forbidden = {"name", "first_name", "last_name", "filename", "file_name", "email", "phone"}

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    ranking = rank(JOB_3Y, [cand("cand-001", BACKEND_3Y)], {"cand-001": 0.5})
    found = set(keys(ranking.model_dump()))
    assert "candidate_id" in found
    assert not found & forbidden


def test_as_of_is_recorded_and_duplicates_rejected():
    assert rank(JOB_3Y, [cand("a", BACKEND_3Y)], {"a": 0.5}).as_of == AS_OF
    with pytest.raises(ValueError):
        rank(JOB_3Y, [cand("a", BACKEND_3Y), cand("a", BACKEND_1Y)], {"a": 0.5})
