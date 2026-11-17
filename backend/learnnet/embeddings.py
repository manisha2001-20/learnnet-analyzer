"""Semantic embeddings: Sentence-Transformers when available, TF-IDF + SVD fallback otherwise."""
from __future__ import annotations

import logging
import os

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = f"sbert:{model_name.split('/')[-1]}"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32))


class TfidfSvdEmbedder(Embedder):
    """Latent semantic analysis: offline, fast, deterministic. Used in tests and when no model can be loaded."""

    name = "tfidf-svd"

    def __init__(self, dims: int = 64):
        self.dims = dims
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.svd = None

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        m = self.vec.fit_transform(texts)
        k = max(2, min(self.dims, m.shape[0] - 1, m.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(m)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("TfidfSvdEmbedder must be fitted first")
        return normalize(self.svd.transform(self.vec.transform(texts)))


def make_embedder(backend: str | None = None) -> Embedder:
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "auto")).lower()
    if backend in ("auto", "sbert"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            if backend == "sbert":
                raise
            log.warning("Sentence-Transformers unavailable (%s); falling back to TF-IDF+SVD", exc)
    return TfidfSvdEmbedder()
