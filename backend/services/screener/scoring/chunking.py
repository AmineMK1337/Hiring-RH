"""Split a sanitized resume into small chunks (one sentence or bullet each).

Small chunks let each job requirement find its own best piece of evidence,
instead of comparing one diluted whole-resume vector to the whole job.
Exact duplicates are dropped and the chunk count is capped, so repeating a
line many times adds nothing.
"""
import re
from dataclasses import dataclass

from services.screener.models import CandidateDocument

_SPLIT = re.compile(r"\n+|(?<=[.!?;])\s+")
_BULLET = re.compile(r"^\s*[-•*▪◦]\s*")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    candidate_id: str
    section: str
    text: str


def split_units(text: str) -> list[str]:
    units = []
    for part in _SPLIT.split(text):
        part = _SPACES.sub(" ", _BULLET.sub("", part)).strip()
        if len(part) >= 3:
            units.append(part)
    return units


def chunk_candidate(
    doc: CandidateDocument, max_chars: int = 400, max_chunks: int = 60
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for section in doc.sections:
        for unit in split_units(section.text):
            text = unit[:max_chars]
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.candidate_id}:{len(chunks)}",
                    candidate_id=doc.candidate_id,
                    section=section.kind,
                    text=text,
                )
            )
            if len(chunks) >= max_chunks:
                return chunks
    return chunks
