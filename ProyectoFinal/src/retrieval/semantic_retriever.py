"""Recuperador semántico con sentence-transformers + similitud coseno."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from .. import config
from .base import BaseRetriever


class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(
        self,
        model_name: str = config.SEMANTIC_MODEL_NAME,
        batch_size: int = config.SEMANTIC_BATCH_SIZE,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.embeddings: np.ndarray | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, corpus: Sequence[str]) -> "SemanticRetriever":
        self.embeddings = self.model.encode(
            list(corpus),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None:
            raise RuntimeError("Llama a fit() antes de search().")
        q_vec = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        # Coseno = producto punto cuando los vectores están normalizados.
        scores = self.embeddings @ q_vec.T
        scores = scores.ravel()
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx, scores[top_idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.embeddings)
        meta = path.with_suffix(".meta")
        meta.write_text(f"{self.model_name}\n{self.batch_size}\n")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticRetriever":
        path = Path(path)
        emb_path = path.with_suffix(".npy")
        meta_path = path.with_suffix(".meta")
        model_name, batch_size = meta_path.read_text().strip().splitlines()
        inst = cls(model_name=model_name, batch_size=int(batch_size))
        inst.embeddings = np.load(emb_path)
        return inst


class HybridRetriever(BaseRetriever):
    """Combinación lineal de dos recuperadores tras min-max normalización."""

    name = "hybrid"

    def __init__(self, retriever_a: BaseRetriever, retriever_b: BaseRetriever, alpha: float = config.HYBRID_ALPHA):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.alpha = alpha

    def fit(self, corpus: Sequence[str]) -> "HybridRetriever":
        self.retriever_a.fit(corpus)
        self.retriever_b.fit(corpus)
        return self

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # Pedimos un top_k grande para tener candidatos comunes.
        k_pool = max(top_k * 5, 50)
        idx_a, scores_a = self.retriever_a.search(query, top_k=k_pool)
        idx_b, scores_b = self.retriever_b.search(query, top_k=k_pool)

        scores_a = self._normalize(scores_a)
        scores_b = self._normalize(scores_b)

        merged: dict[int, float] = {}
        for i, s in zip(idx_a, scores_a):
            merged[int(i)] = merged.get(int(i), 0.0) + self.alpha * float(s)
        for i, s in zip(idx_b, scores_b):
            merged[int(i)] = merged.get(int(i), 0.0) + (1 - self.alpha) * float(s)

        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]
        indices = np.array([i for i, _ in ranked], dtype=int)
        scores = np.array([s for _, s in ranked], dtype=float)
        return indices, scores

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Guarda cada recuperador por separado.")

    @classmethod
    def load(cls, path: str | Path) -> "HybridRetriever":
        raise NotImplementedError("Carga cada recuperador por separado.")
