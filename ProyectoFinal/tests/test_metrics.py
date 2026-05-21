"""Pruebas de métricas de evaluación."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_perfect():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_at_k_none():
    assert precision_at_k([10, 11, 12], {1, 2, 3}, k=3) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 2, 99], {1, 2, 3}, k=3) == 2 / 3


def test_average_precision_textbook():
    # Relevantes en posiciones 1 y 3: AP = (1/1 + 2/3) / 2 ≈ 0.833
    ap = average_precision([10, 20, 30, 40], {10, 30})
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9


def test_map_aggregates_correctly():
    retrieved = [[1, 2, 3], [4, 5, 6]]
    relevant = [{1}, {6}]
    # AP1 = 1.0, AP2 = 1/3
    assert abs(mean_average_precision(retrieved, relevant) - (1.0 + 1 / 3) / 2) < 1e-9


def test_ndcg_at_k_bounded():
    score = ndcg_at_k([1, 2, 3], {1, 2}, k=3)
    assert 0.0 <= score <= 1.0
