"""Pruebas de los recuperadores TF-IDF y BM25 (semántico se omite por costo)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import BM25Retriever, TfidfRetriever


CORPUS = [
    "vaccines cause autism in children according to fake report",
    "covid vaccine reduces hospitalization rates significantly",
    "trump claims election was rigged with no evidence",
    "climate change is supported by scientific consensus globally",
    "celebrity diet plan promises to cure cancer in days",
    "nasa confirms water on the surface of the moon",
]


@pytest.mark.parametrize("RetrieverCls", [TfidfRetriever, BM25Retriever])
def test_retriever_returns_top_k(RetrieverCls):
    retriever = RetrieverCls().fit(CORPUS)
    idx, scores = retriever.search("vaccine children", top_k=3)
    assert len(idx) == 3
    assert len(scores) == 3
    assert all(0 <= i < len(CORPUS) for i in idx)


@pytest.mark.parametrize("RetrieverCls", [TfidfRetriever, BM25Retriever])
def test_retriever_scores_are_descending(RetrieverCls):
    retriever = RetrieverCls().fit(CORPUS)
    _, scores = retriever.search("election trump", top_k=4)
    diffs = np.diff(scores)
    assert (diffs <= 1e-9).all(), f"Scores no decrecientes: {scores}"


@pytest.mark.parametrize("RetrieverCls", [TfidfRetriever, BM25Retriever])
def test_retriever_finds_relevant(RetrieverCls):
    retriever = RetrieverCls().fit(CORPUS)
    idx, _ = retriever.search("nasa moon water", top_k=1)
    assert idx[0] == 5


def test_search_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfRetriever().search("x", top_k=1)
    with pytest.raises(RuntimeError):
        BM25Retriever().search("x", top_k=1)
