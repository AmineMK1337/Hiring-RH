# Configuration for the screener service
"""Screener settings. Values can be overridden with environment variables."""
import math
import os
from dataclasses import dataclass, field

import numpy as np

EMBEDDING_MODEL = os.getenv("SCREENER_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


def _default_weights() -> dict[str, float]:
    return {"required": 2.0, "preferred": 1.0, "responsibility": 1.0}


@dataclass(frozen=True)
class SemanticConfig:
    # Raw cosine depends on the model: for E5 even unrelated text scores
    # around 0.7-0.8, so [floor, ceiling] is mapped to [0, 1].
    # Calibrate on your own data before trusting the numbers (see README).
    floor: float = float(os.getenv("SCREENER_SEMANTIC_FLOOR", "0.78"))
    ceiling: float = float(os.getenv("SCREENER_SEMANTIC_CEILING", "0.90"))
    met_threshold: float = 0.5           # calibrated score counted as "met"
    max_chunk_chars: int = 400
    max_chunks_per_candidate: int = 60
    weights: dict[str, float] = field(default_factory=_default_weights)

    def __post_init__(self) -> None:
        if self.ceiling <= self.floor:
            raise ValueError("ceiling must be greater than floor")

    def calibrate(self, sims: np.ndarray) -> np.ndarray:
        return np.clip((sims - self.floor) / (self.ceiling - self.floor), 0.0, 1.0)


@dataclass(frozen=True)
class RankingWeights:
    # Weights of the combined score. A criterion that does not apply to a
    # candidate (score None) is left out and the others are renormalized.
    semantic: float = float(os.getenv("SCREENER_WEIGHT_SEMANTIC", "0.5"))
    experience: float = float(os.getenv("SCREENER_WEIGHT_EXPERIENCE", "0.2"))
    graph: float = float(os.getenv("SCREENER_WEIGHT_GRAPH", "0.3"))

    def __post_init__(self) -> None:
        values = self.as_dict().values()
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("ranking weights must be finite and non-negative")
        if not any(v > 0 for v in values):
            raise ValueError("at least one ranking weight must be positive")

    def as_dict(self) -> dict[str, float]:
        return {"semantic": self.semantic, "experience": self.experience, "graph": self.graph}
