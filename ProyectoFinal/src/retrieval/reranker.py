"""Reranker cross-encoder para retrieval en dos etapas.

Inspirado en Nogueira & Cho (2019) *Passage Re-ranking with BERT*. Patrón:

    BM25 (rápido) → top-100 candidatos → CrossEncoder (preciso) → top-K final

Un cross-encoder ve la consulta y el documento juntos en la misma entrada
de BERT, lo que captura interacciones imposibles con dos torres separadas.
Es más caro pero solo se aplica sobre un puñado de candidatos.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .. import config
from .base import BaseRetriever


class CrossEncoderReranker:
    """Wrapper sobre `sentence-transformers/cross-encoder`.

    Modelos recomendados:
    - ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — rápido, 22M params, score 0-10
    - ``cross-encoder/ms-marco-MiniLM-L-12-v2`` — más preciso, +50% lento
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length,
            )
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve `(indices_relativos, scores)` ordenados por relevancia desc.

        ``indices_relativos`` son posiciones dentro de la lista ``documents``,
        no índices del corpus original; eso lo maneja ``RerankedRetriever``.
        """
        if not documents:
            return np.array([], dtype=int), np.array([], dtype=float)

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = np.asarray(scores, dtype=float)

        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return order, scores[order]


class RerankedRetriever(BaseRetriever):
    """Pipeline two-stage: recuperador base + reranker.

    Args
    ----
    base : BaseRetriever
        Recuperador rápido (BM25 o TF-IDF) que filtra a un pool grande.
    reranker : CrossEncoderReranker
        Modelo que reordena los candidatos con atención query↔doc.
    candidate_pool : int
        Tamaño del pool intermedio. Más alto = mejor recall pero más lento.
    documents : list[str] | None
        Corpus en texto plano (necesario para que el reranker pueda leer los docs).
        Si es ``None``, se intentará tomar del recuperador base.
    """

    name = "reranked"

    def __init__(
        self,
        base: BaseRetriever,
        reranker: CrossEncoderReranker | None = None,
        candidate_pool: int = 100,
        documents: Sequence[str] | None = None,
    ):
        self.base = base
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_pool = candidate_pool
        self.documents: list[str] | None = list(documents) if documents is not None else None

    def fit(self, corpus: Sequence[str]) -> "RerankedRetriever":
        self.documents = list(corpus)
        self.base.fit(corpus)
        return self

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.documents is None:
            raise RuntimeError("Llama a fit() antes de search() o pasa `documents` al constructor.")

        # Etapa 1: candidatos rápidos.
        pool = max(self.candidate_pool, top_k)
        cand_idx, _cand_scores = self.base.search(query, top_k=pool)

        # Etapa 2: reranking sobre los textos completos de los candidatos.
        cand_docs = [self.documents[int(i)] for i in cand_idx]
        rel_order, rerank_scores = self.reranker.rerank(query, cand_docs, top_k=top_k)

        final_indices = cand_idx[rel_order]
        return final_indices, rerank_scores

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Guarda el recuperador base por separado; el reranker no requiere persistencia.")

    @classmethod
    def load(cls, path: str | Path) -> "RerankedRetriever":
        raise NotImplementedError("Reconstruye el RerankedRetriever cargando el base manualmente.")
