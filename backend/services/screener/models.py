# Data models used by the screener service
"""Data contracts for semantic matching.

Candidates arrive here already sanitized: pseudonymous id, visible text only,
no name, email, phone, photo, age or address. Hidden text must be stripped
upstream, because semantic similarity is exactly what keyword stuffing targets.
"""
from typing import Literal

from pydantic import BaseModel, Field

RequirementKind = Literal["required", "preferred", "responsibility"]
SectionKind = Literal["summary", "experience", "education", "project", "skills", "other"]


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
