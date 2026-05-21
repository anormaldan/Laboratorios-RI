"""Recuperador basado en TF-IDF + similitud coseno."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import config
from ..preprocessing import clean_text
from .base import BaseRetriever


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, **vectorizer_kwargs):
        params = {**config.TFIDF_PARAMS, **vectorizer_kwargs}
        self.vectorizer = TfidfVectorizer(**params)
        self.matrix = None

    def fit(self, corpus: Sequence[str]) -> "TfidfRetriever":
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    def transform_query(self, query: str):
        return self.vectorizer.transform([clean_text(query)])

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.matrix is None:
            raise RuntimeError("Llama a fit() antes de search().")
        q_vec = self.transform_query(query)
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        # argpartition + ordenamiento parcial: O(n) en vez de O(n log n)
        k = min(top_k, sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return top_idx, sims[top_idx]

    def save(self, path: str | Path) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfRetriever":
        data = joblib.load(path)
        inst = cls()
        inst.vectorizer = data["vectorizer"]
        inst.matrix = data["matrix"]
        return inst
