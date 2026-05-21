"""Verificador de afirmaciones inspirado en FEVER (Thorne et al., 2018).

A diferencia del clasificador binario fake/real, este módulo evalúa la relación
entre una afirmación (claim) y un conjunto de documentos de evidencia,
produciendo una distribución sobre 3 clases:

    - SUPPORTS  : la evidencia respalda la afirmación
    - REFUTES   : la evidencia contradice la afirmación
    - NEI       : Not Enough Information

Usa modelos NLI pre-entrenados (entailment, contradiction, neutral).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


# Mapeo de etiquetas NLI estándar → FEVER
NLI_TO_FEVER = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "NEI",
}


@dataclass
class VerificationResult:
    verdict: str                         # SUPPORTS | REFUTES | NEI
    scores: Dict[str, float]             # distribución sobre las 3 clases
    per_evidence: List[Dict[str, float]] # score por documento de evidencia
    aggregation: str = "max_supports_or_refutes"

    def __str__(self) -> str:
        return (
            f"{self.verdict} (S={self.scores['SUPPORTS']:.2f} "
            f"R={self.scores['REFUTES']:.2f} N={self.scores['NEI']:.2f})"
        )


class EvidenceVerifier:
    """Verificador NLI sobre pares (claim, evidence).

    Args
    ----
    model_name : str
        Cross-encoder NLI compatible con sentence-transformers.
        Recomendado:
        - ``cross-encoder/nli-deberta-v3-base`` (más preciso)
        - ``cross-encoder/nli-MiniLM2-L6-H768`` (más ligero)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str | None = None,
        batch_size: int = 16,
        max_evidence_chars: int = 800,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_evidence_chars = max_evidence_chars
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _predict_nli(self, claim: str, evidences: Sequence[str]) -> np.ndarray:
        """Devuelve matriz (n_evidence, 3) con probabilidades [entail, contradict, neutral]."""
        pairs = [(ev[: self.max_evidence_chars], claim) for ev in evidences]
        # sentence-transformers CrossEncoder predice logits, aplicamos softmax.
        logits = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        # Softmax estable por fila.
        logits = np.asarray(logits, dtype=float)
        if logits.ndim == 1:
            logits = logits[None, :]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return probs

    def verify(
        self,
        claim: str,
        evidences: Sequence[str],
        threshold: float = 0.6,
    ) -> VerificationResult:
        """Verifica una afirmación contra una lista de documentos de evidencia.

        Estrategia de agregación (siguiendo FEVER 2018):
        1. Calcular probas por documento.
        2. Si algún documento da SUPPORTS o REFUTES con prob > threshold,
           gana la etiqueta más confiada.
        3. Si ninguno supera el umbral → NEI.
        """
        if not evidences:
            return VerificationResult(
                verdict="NEI",
                scores={"SUPPORTS": 0.0, "REFUTES": 0.0, "NEI": 1.0},
                per_evidence=[],
            )

        probs = self._predict_nli(claim, evidences)  # cols: [entail, contradict, neutral]

        # Score por evidencia, ya mapeado a FEVER
        per_evidence: List[Dict[str, float]] = [
            {
                "SUPPORTS": float(p[0]),
                "REFUTES": float(p[1]),
                "NEI": float(p[2]),
            }
            for p in probs
        ]

        # Agregado: max sobre las evidencias.
        max_support = float(probs[:, 0].max())
        max_refute = float(probs[:, 1].max())
        avg_neutral = float(probs[:, 2].mean())

        agg = {"SUPPORTS": max_support, "REFUTES": max_refute, "NEI": avg_neutral}

        if max_support >= threshold and max_support >= max_refute:
            verdict = "SUPPORTS"
        elif max_refute >= threshold:
            verdict = "REFUTES"
        else:
            verdict = "NEI"

        return VerificationResult(verdict=verdict, scores=agg, per_evidence=per_evidence)

    def explain(self, claim: str, evidences: Sequence[str], top_k: int = 3) -> List[Dict]:
        """Devuelve los pasajes de evidencia que más explican el veredicto.

        Útil para mostrar al usuario *por qué* el sistema dio cierto resultado.
        """
        probs = self._predict_nli(claim, evidences)
        # Score = max(entailment, contradiction) → cuán informativo es el pasaje.
        informativeness = np.maximum(probs[:, 0], probs[:, 1])
        order = np.argsort(-informativeness)[:top_k]

        return [
            {
                "evidence": evidences[int(i)][:300],
                "supports": float(probs[i, 0]),
                "refutes": float(probs[i, 1]),
                "informativeness": float(informativeness[i]),
            }
            for i in order
        ]
