"""Interfaz común para todos los recuperadores."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class BaseRetriever(ABC):
    """Contrato compartido por TF-IDF, BM25 y Semantic."""

    name: str = "base"

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> "BaseRetriever":
        """Indexa el corpus. Debe almacenar lo necesario para `search`."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (indices, scores) ordenados por relevancia descendente."""

    def search_batch(
        self, queries: Sequence[str], top_k: int = 10
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Búsqueda secuencial sobre múltiples consultas (override para paralelizar)."""
        return [self.search(q, top_k=top_k) for q in queries]

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRetriever": ...
