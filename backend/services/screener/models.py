# Data models used by the screener service
"""Data contracts for semantic matching and the combined ranking.

Candidates arrive here already sanitized: pseudonymous id, visible text only,
no name, email, phone, photo, age or address. Hidden text must be stripped
upstream, because semantic similarity is exactly what keyword stuffing targets.
Outputs identify candidates by candidate_id only.
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

RequirementKind = Literal["required", "preferred", "responsibility"]
SectionKind = Literal["summary", "experience", "education", "project", "skills", "other"]
Criterion = Literal["semantic", "experience", "graph"]


class Requirement(BaseModel):
    text: str = Field(min_length=3)
    kind: RequirementKind = "required"


class JobOffer(BaseModel):
    job_id: str
    title: str
    requirements: list[Requirement] = Field(min_length=1)


class ResumeSection(BaseModel):
    kind: SectionKind
    text: str


class CandidateDocument(BaseModel):
    candidate_id: str            # pseudonym, never the real name
    sections: list[ResumeSection]
    injection_flag: bool = False  # set by the sanitization pipeline


class RequirementEvidence(BaseModel):
    requirement: str
    kind: RequirementKind
    score: float                 # calibrated, 0..1
    raw_similarity: float        # cosine, for debugging and calibration
    best_chunk_id: str | None    # resolve with SemanticIndex.chunk_text()


class SemanticMatch(BaseModel):
    candidate_id: str
    rank: int
    score: float                 # weighted mean of calibrated scores, 0..1
    requirements_met: int
    requirements_total: int
    injection_flag: bool
    evidence: list[RequirementEvidence]


class SemanticRanking(BaseModel):
    """One ranking run. Store it as-is so the result can be reproduced and audited."""
    job_id: str
    model_id: str
    config: dict
    matches: list[SemanticMatch]


class ExperienceResult(BaseModel):
    """Job-relevant experience, computed from dated positions in experience sections."""
    relevant_years: float         # in positions that mention the job's skills or role
    total_years: float            # all dated positions, overlaps counted once
    required_years: float | None  # None = the job states no requirement
    score: float | None           # 0..1, None = not applicable (no requirement)
    positions_dated: int
    positions_relevant: int
    explanation: str


class CriterionScores(BaseModel):
    """Per-criterion scores in 0..1. None = not applicable, left out of the total."""
    semantic: float | None
    experience: float | None
    graph: float | None


class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    total_score: float                 # weighted mean over applied criteria, 0..1
    scores: CriterionScores            # kept separate so large gaps can be flagged later
    applied_criteria: list[Criterion]
    weights_used: dict[str, float]     # renormalized over applied criteria, sums to 1
    matched_skills: list[str]          # display only: job skills found in the resume
    missing_skills: list[str]          # display only: job skills not found
    additional_skills: list[str]       # display only, never scored
    experience: ExperienceResult
    injection_flag: bool
    requirements_met: int
    requirements_total: int
    evidence: list[RequirementEvidence]


class CombinedRanking(BaseModel):
    """One combined ranking run. Store it as-is so it can be reproduced and audited."""
    job_id: str
    as_of: date                        # reference date for "present" in resumes
    required_years: float | None       # parsed from the job, None = no requirement
    weights_config: dict[str, float]   # configured weights, before renormalization
    semantic_model_id: str
    semantic_config: dict
    graph_scorer: str
    candidates: list[RankedCandidate]
