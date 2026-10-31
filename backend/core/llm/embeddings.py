"""Shared embedding backends, used by the screener and employee_qa services.

Requirements are embedded as *queries* and resume chunks as *passages*:
asymmetric retrieval models (E5, BGE) are trained that way and score better.
All vectors are L2-normalized, so a dot product is a cosine similarity.
"""
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_id: str

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local model, so candidate text never leaves the machine.

    Default is multilingual because resumes may be in English, French or Arabic.
    E5 models expect "query: " / "passage: " prefixes.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_id = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._query_prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._passage_prefix + t for t in texts])


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests and offline demos.

    Not semantic (no synonyms), but needs no model download.
    """

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dim: int = 1024):
        self.model_id = f"hashing-bow-{dim}"
        self._dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in self._TOKEN.findall(text.casefold()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "little") % self._dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)
