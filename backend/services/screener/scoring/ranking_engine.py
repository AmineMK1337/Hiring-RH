# Ranks candidates against a job offer
"""Combined ranking: semantic + job-relevant experience + skill graph.

Each criterion gives a score in 0..1 or None (not applicable: the job states no
experience requirement, or the graph scorer is not available). None criteria
are left out and the remaining weights are renormalized, so a candidate is
never penalized for a criterion the job does not ask for.

The engine ranks, flags and explains. It never shortlists, rejects or sends
anything: a recruiter decides, through the approval queue.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from services.screener.config import RankingWeights
from services.screener.graph_store.taxonomy import SkillTaxonomy, default_taxonomy
from services.screener.models import (
    CandidateDocument,
    CombinedRanking,
    CriterionScores,
    JobOffer,
    RankedCandidate,
    SemanticRanking,
)
from services.screener.scoring.experience import parse_required_years, relevant_experience
from services.screener.scoring.normalize import fold, term_pattern

CRITERIA = ("semantic", "experience", "graph")

# Role names that make a position relevant even without a skill mention. A
# group applies when one of its terms appears in the job; then all its terms
# (English, French, Arabic) count. Generic words alone ("developer",
# "engineer", "manager") are left out on purpose: "business developer" is sales.
ROLE_GROUPS: dict[str, list[str]] = {
    "backend": ["backend", "back-end", "back end", "الواجهة الخلفية"],
    "frontend": ["frontend", "front-end", "front end", "الواجهة الأمامية"],
    "full stack": ["full stack", "full-stack", "fullstack"],
    "devops": ["devops", "dev ops", "site reliability"],
    "data analyst": ["data analyst", "analyste de données", "analyste data", "محلل بيانات"],
    "data scientist": ["data scientist", "scientifique des données", "عالم بيانات"],
    "data engineer": ["data engineer", "ingénieur de données", "ingénieur data", "مهندس بيانات"],
    "machine learning engineer": ["machine learning engineer", "ml engineer",
                                  "ingénieur machine learning"],
    "software engineer": ["software engineer", "software developer", "ingénieur logiciel",
                          "développeur logiciel", "مهندس برمجيات", "مطور برمجيات"],
}


class SemanticRanker(Protocol):
    """What the engine needs from SemanticIndex."""

    def rank(self, job: JobOffer, top_k: int | None = None) -> SemanticRanking: ...


@dataclass(frozen=True)
class GraphResult:
    score: float | None          # 0..1, None = not applicable
    detail: str = ""


class GraphScorer(Protocol):
    """Skill-graph score (Neo4j). Plugs in later without changes to the engine."""

    def score(self, job: JobOffer, candidate_ids: list[str]) -> dict[str, GraphResult]: ...


class NullGraphScorer:
    """Used until the Neo4j scorer exists: the graph criterion is not applied."""

    def score(self, job: JobOffer, candidate_ids: list[str]) -> dict[str, GraphResult]:
        return {cid: GraphResult(None, "graph scorer not available") for cid in candidate_ids}


def job_text(job: JobOffer) -> str:
    return "\n".join([job.title, *(req.text for req in job.requirements)])


def role_terms(text: str) -> set[str]:
    folded = fold(text)
    terms: set[str] = set()
    for group in ROLE_GROUPS.values():
        pattern = term_pattern(group)
        if pattern is not None and pattern.search(folded):
            terms.update(group)
    return terms


def combine(
    scores: dict[str, float | None], weights: RankingWeights
) -> tuple[float, list[str], dict[str, float]]:
    """Weighted mean over applicable criteria: (total, applied, renormalized weights)."""
    configured = weights.as_dict()
    applied = [c for c in CRITERIA if scores[c] is not None and configured[c] > 0]
    weight_sum = sum(configured[c] for c in applied)
    if weight_sum == 0:
        return 0.0, [], {}
    used = {c: configured[c] / weight_sum for c in applied}
    return sum(used[c] * scores[c] for c in applied), applied, used


def rank_combined(
    job: JobOffer,
    candidates: Iterable[CandidateDocument],
    semantic: SemanticRanker,
    *,
    as_of: date,
    graph_scorer: GraphScorer | None = None,
    weights: RankingWeights | None = None,
    taxonomy: SkillTaxonomy | None = None,
) -> CombinedRanking:
    """Rank the given candidates for one job.

    `semantic` must already contain these candidates (SemanticIndex.upsert).
    `as_of` is the reference date for "present" in resumes; pass a fixed date
    to reproduce a ranking. Order: total score, then semantic score (both
    descending), then candidate_id.
    """
    graph_scorer = graph_scorer or NullGraphScorer()
    weights = weights or RankingWeights()
    taxonomy = taxonomy or default_taxonomy()

    docs = list(candidates)
    ids = [doc.candidate_id for doc in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate_id in candidates")

    text = job_text(job)
    required_years = parse_required_years(text)
    job_skills = taxonomy.find_skills(text)
    relevance_terms = taxonomy.terms_for(job_skills) | role_terms(text)

    semantic_ranking = semantic.rank(job)
    semantic_matches = {m.candidate_id: m for m in semantic_ranking.matches}
    graph_results = graph_scorer.score(job, ids)

    rows = []
    for doc in docs:
        match = semantic_matches.get(doc.candidate_id)
        graph = graph_results.get(doc.candidate_id)
        if graph is not None and graph.score is not None and not 0 <= graph.score <= 1:
            raise ValueError(f"graph score out of range for {doc.candidate_id}")
        experience = relevant_experience(
            doc.sections, relevance_terms, required_years=required_years, as_of=as_of
        )
        scores = {
            "semantic": match.score if match else None,
            "experience": experience.score,
            "graph": graph.score if graph else None,
        }
        total, applied, used = combine(scores, weights)
        candidate_skills = taxonomy.find_skills("\n".join(s.text for s in doc.sections))
        rows.append((round(total, 4), doc, match, scores, applied, used, candidate_skills, experience))

    rows.sort(key=lambda r: (-r[0], -(r[3]["semantic"] or 0.0), r[1].candidate_id))

    ranked = [
        RankedCandidate(
            candidate_id=doc.candidate_id,
            rank=position,
            total_score=total,
            scores=CriterionScores(**scores),
            applied_criteria=applied,
            weights_used={c: round(w, 4) for c, w in used.items()},
            matched_skills=sorted(job_skills & skills),
            missing_skills=sorted(job_skills - skills),
            additional_skills=sorted(skills - job_skills),
            experience=experience,
            injection_flag=doc.injection_flag,
            requirements_met=match.requirements_met if match else 0,
            requirements_total=len(job.requirements),
            evidence=match.evidence if match else [],
        )
        for position, (total, doc, match, scores, applied, used, skills, experience)
        in enumerate(rows, start=1)
    ]
    return CombinedRanking(
        job_id=job.job_id,
        as_of=as_of,
        required_years=required_years,
        weights_config=weights.as_dict(),
        semantic_model_id=semantic_ranking.model_id,
        semantic_config=semantic_ranking.config,
        graph_scorer=type(graph_scorer).__name__,
        candidates=ranked,
    )
