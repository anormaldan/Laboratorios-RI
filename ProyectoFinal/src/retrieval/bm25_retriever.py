"""Recuperador basado en BM25 (rank-bm25)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np
from rank_bm25 import BM25Okapi

from .. import config
from ..preprocessing import tokenize
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, k1: float = config.BM25_PARAMS["k1"], b: float = config.BM25_PARAMS["b"]):
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] | None = None

    def fit(self, corpus: Sequence[str]) -> "BM25Retriever":
        self.tokenized_corpus = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.bm25 is None:
            raise RuntimeError("Llama a fit() antes de search().")
        q_tokens = tokenize(query)
        scores = np.asarray(self.bm25.get_scores(q_tokens))
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx, scores[top_idx]

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus,
             "k1": self.k1, "b": self.b},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        data = joblib.load(path)
        inst = cls(k1=data["k1"], b=data["b"])
        inst.bm25 = data["bm25"]
        inst.tokenized_corpus = data["tokenized_corpus"]
        return inst
