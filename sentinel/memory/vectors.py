"""Embedding backends and a small in-process vector index.

The long-term memory needs similarity search over short text records
(alert signatures, human correction notes, investigation conclusions). That is
a few thousand rows at most, so a numpy brute-force cosine scan is both faster
and far simpler than adding a vector database.

Two backends:
  HashEmbedder -- deterministic, dependency-free, no model. Good enough to
      cluster near-identical alert signatures, and it makes tests exact.
  NVRCLIPEmbedder -- delegates to NVR's own embedding search over events, so
      we reuse the CLIP index that is already maintained instead of standing
      up a second one.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


class HashEmbedder(Embedder):
    """Hashed bag-of-words with character trigrams, L2-normalized.

    Deterministic across processes and runs -- which matters, because the
    pattern library is persisted and must stay comparable after a restart.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        # Character trigrams give partial credit for near-miss wording.
        joined = " ".join(tokens)
        feats += [joined[i : i + 3] for i in range(max(0, len(joined) - 2))]
        return feats

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in self._features(text):
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


class NVRCLIPEmbedder(Embedder):
    """Placeholder that routes text similarity through NVR's search API.

    NVR does not expose raw embedding vectors over HTTP, only ranked search
    results, so this backend cannot produce a vector. It exists so the config
    surface is honest about the option; ``embed`` raises, and callers that
    want NVR similarity should use ``tools.similar_alerts`` (which calls
    /events/search) instead of the local index.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "NVR's CLIP index is only reachable as ranked search; use the "
            "search_similar_alerts tool for NVR-side similarity."
        )


def build_embedder(backend: str, dim: int) -> Embedder:
    if backend == "hash":
        return HashEmbedder(dim)
    if backend == "nvr_clip":
        return NVRCLIPEmbedder(dim)
    raise ValueError(f"unknown embedding backend: {backend}")


def cosine_search(
    query: np.ndarray, matrix: np.ndarray, top_k: int = 5
) -> list[tuple[int, float]]:
    """Return (row_index, similarity) pairs, best first.

    Vectors are assumed L2-normalized, so the dot product is the cosine.
    """
    if matrix.size == 0:
        return []
    sims = matrix @ query
    k = min(top_k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def decay_factor(age_days: float, half_life_days: float) -> float:
    """Exponential decay so patterns that stop recurring lose influence."""
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, max(0.0, age_days) / half_life_days)
