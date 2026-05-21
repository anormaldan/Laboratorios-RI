"""Clasificador binario fake/real entrenado sobre vectores TF-IDF."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .. import config
from ..preprocessing import clean_text


def _build_estimator(kind: str):
    if kind == "logreg":
        return LogisticRegression(
            max_iter=1000, n_jobs=-1, C=4.0, random_state=config.SEED,
        )
    if kind == "nb":
        return MultinomialNB(alpha=0.3)
    raise ValueError(f"Tipo de clasificador desconocido: {kind}")


class FakeNewsClassifier:
    """Pipeline TF-IDF + (LogReg | NB) con API homogénea."""

    def __init__(self, kind: str = config.CLASSIFIER_TYPE):
        self.kind = kind
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS)),
            ("clf", _build_estimator(kind)),
        ])

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "FakeNewsClassifier":
        cleaned = [clean_text(t) for t in texts]
        self.pipeline.fit(cleaned, labels)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        return self.pipeline.predict(cleaned)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Devuelve P(real) para cada texto."""
        cleaned = [clean_text(t) for t in texts]
        proba = self.pipeline.predict_proba(cleaned)
        # Columna 1 = clase REAL (label=1).
        return proba[:, 1]

    def evaluate(self, texts: Sequence[str], labels: Sequence[int]) -> dict:
        preds = self.predict(texts)
        return {
            "report": classification_report(labels, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }

    def save(self, path: str | Path = config.CLASSIFIER_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: str | Path = config.CLASSIFIER_PATH) -> "FakeNewsClassifier":
        data = joblib.load(path)
        inst = cls(kind=data["kind"])
        inst.pipeline = data["pipeline"]
        return inst
