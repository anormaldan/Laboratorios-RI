"""Métricas de evaluación de recuperación de información.

Implementaciones didácticas y autocontenidas para defender en exposición.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def reciprocal_rank(retrieved: Sequence[int], relevant: Iterable[int]) -> float:
    """Recíproco del rango del primer relevante (0 si no hay ninguno)."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: Sequence[Sequence[int]],
    relevant_lists: Sequence[Iterable[int]],
) -> float:
    if not retrieved_lists:
        return 0.0
    return float(np.mean([
        reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, relevant_lists)
    ]))


def precision_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fracción de los top-K que son relevantes."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant_set) / k


def recall_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fracción de los documentos relevantes que aparecen en los top-K."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / len(relevant_set)


def average_precision(retrieved: Sequence[int], relevant: Iterable[int]) -> float:
    """Promedio de Precision@k en cada posición donde aparece un relevante."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            score += hits / i
    return score / len(relevant_set)


def mean_average_precision(
    retrieved_lists: Sequence[Sequence[int]],
    relevant_lists: Sequence[Iterable[int]],
) -> float:
    if not retrieved_lists:
        return 0.0
    return float(np.mean([
        average_precision(r, rel) for r, rel in zip(retrieved_lists, relevant_lists)
    ]))


def dcg_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """DCG con ganancias binarias (1 si relevante, 0 si no)."""
    relevant_set = set(relevant)
    gains = [1.0 if d in relevant_set else 0.0 for d in retrieved[:k]]
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(
    retriever,
    queries: Sequence[str],
    ground_truth: Sequence[Iterable[int]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """Devuelve un dict con métricas agregadas para un recuperador dado."""
    retrieved_all = []
    max_k = max(k_values)
    for q in queries:
        idx, _ = retriever.search(q, top_k=max_k)
        retrieved_all.append(list(map(int, idx)))

    results: dict = {"name": getattr(retriever, "name", retriever.__class__.__name__)}
    for k in k_values:
        results[f"P@{k}"] = float(np.mean([
            precision_at_k(r, g, k) for r, g in zip(retrieved_all, ground_truth)
        ]))
        results[f"R@{k}"] = float(np.mean([
            recall_at_k(r, g, k) for r, g in zip(retrieved_all, ground_truth)
        ]))
        results[f"NDCG@{k}"] = float(np.mean([
            ndcg_at_k(r, g, k) for r, g in zip(retrieved_all, ground_truth)
        ]))
    results["MAP"] = mean_average_precision(retrieved_all, ground_truth)
    results["MRR"] = mean_reciprocal_rank(retrieved_all, ground_truth)
    return results
