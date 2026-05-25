"""Clasificador binario basado en Transformers (ALBERT / DistilBERT).

Motivación científica
---------------------
* Sutradhar et al. (2023) — *Machine Learning Technique Based Fake News Detection*
  (Paper 1 del equipo VERITAS): TF-IDF + NB/LR alcanza solo **56 % accuracy** en
  datasets pequeños e imbalanced (F1-macro 32 %).  El modelo aprende artefactos
  de la distribución (frecuencias de palabras) en lugar de semántica real.

* Azizah et al. (2023) — *Performance Analysis of Transformer Based Models*
  (Paper 2 del equipo VERITAS): ALBERT supera a BERT y RoBERTa con **87.6 %**
  accuracy, **86.9 % F1** y solo 174.5 s/época.  Procesa texto en paralelo con
  auto-atención, produciendo representaciones contextuales que capturan matices
  imposibles para TF-IDF.

Esta clase provee la **misma API pública** que ``FakeNewsClassifier``
(fit / predict / predict_proba / evaluate / save / load), por lo que es un
reemplazo directo en ``SearchPipeline.build(use_transformer_clf=True)``.

Instalación de dependencias adicionales::

    pip install torch transformers

Uso rápido::

    clf = TransformerClassifier()          # albert-base-v2 por defecto
    clf.fit(train_texts, train_labels)
    proba = clf.predict_proba(test_texts)  # P(REAL) ∈ [0, 1]
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

_INSTALL_MSG = (
    "TransformerClassifier requiere PyTorch y HuggingFace Transformers.\n"
    "Instálalos con:  pip install torch transformers"
)


# ---------------------------------------------------------------------------
# Dataset interno
# ---------------------------------------------------------------------------

class _NewsDataset:
    """Dataset mínimo compatible con torch.utils.data.Dataset."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        import torch
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Clasificador principal
# ---------------------------------------------------------------------------

class TransformerClassifier:
    """Fine-tuner de transformer binario (FAKE = 0 / REAL = 1).

    Parámetros
    ----------
    model_name : str
        Identificador HuggingFace del modelo base.  Recomendados:

        - ``albert-base-v2``            — 11 M params, Paper 2 winner (default)
        - ``distilbert-base-uncased``   — 66 M params, 40 % más rápido que BERT
        - ``bert-base-uncased``         — baseline clásico de referencia

    max_length : int
        Tokens máximos por documento.  256 captura titular + primer párrafo
        de la gran mayoría de artículos de noticias.
    batch_size : int
        Tamaño de batch.  Reducir a 8 si hay OOM en GPU con poca VRAM.
    num_epochs : int
        Épocas de fine-tuning.  2-3 son suficientes para dominios in-distribution.
    learning_rate : float
        Tasa de Adam.  2e-5 es el estándar para fine-tuning BERT-like
        (Devlin et al., 2019).
    warmup_ratio : float
        Fracción de pasos dedicados al calentamiento lineal del lr.
    """

    def __init__(
        self,
        model_name: str = config.TRANSFORMER_MODEL_NAME,
        device: str | None = None,
        max_length: int = config.TRANSFORMER_MAX_LENGTH,
        batch_size: int = config.TRANSFORMER_BATCH_SIZE,
        num_epochs: int = config.TRANSFORMER_EPOCHS,
        learning_rate: float = config.TRANSFORMER_LR,
        warmup_ratio: float = 0.06,
    ):
        self.model_name = model_name
        self._device_str = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Propiedades lazy
    # ------------------------------------------------------------------

    @property
    def device(self):
        try:
            import torch
        except ImportError:
            raise ImportError(_INSTALL_MSG)
        if self._device_str:
            return torch.device(self._device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(_INSTALL_MSG)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _load_base_model(self):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(_INSTALL_MSG)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        return model.to(self.device)

    # ------------------------------------------------------------------
    # Tokenización
    # ------------------------------------------------------------------

    def _encode(self, texts: Sequence[str]) -> dict:
        """Tokeniza una lista de textos y devuelve tensores PyTorch."""
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        val_texts: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[int]] = None,
    ) -> "TransformerClassifier":
        """Fine-tunea el transformer sobre el corpus de noticias.

        Aplica *class weighting* para manejar desequilibrios FAKE/REAL.
        Usa *gradient clipping* y *linear warmup* (estándar en BERT fine-tuning).

        Parámetros
        ----------
        val_texts / val_labels : opcional
            Si se proveen, se reporta accuracy de validación al final de cada época.
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import get_linear_schedule_with_warmup
        except ImportError:
            raise ImportError(_INSTALL_MSG)

        # Tokenización por chunks para evitar OOM en listas grandes
        chunk = 512
        input_ids_list, attn_list = [], []
        for start in range(0, len(texts), chunk):
            batch = list(texts[start: start + chunk])
            enc = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"])
            attn_list.append(enc["attention_mask"])

        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attn_list, dim=0)
        label_tensor = torch.tensor(list(labels), dtype=torch.long)

        dataset = _NewsDataset(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            list(labels),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Pesos de clase para dataset imbalanced
        counts = np.bincount(list(labels), minlength=2).astype(float)
        weights = 1.0 / (counts + 1e-9)
        weights = weights / weights.sum() * 2
        class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        self._model = self._load_base_model()
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, eps=1e-8
        )

        total_steps = len(loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        logger.info(
            "Fine-tuning %s en %d ejemplos por %d épocas (device=%s)…",
            self.model_name, len(labels), self.num_epochs, self.device
        )

        for epoch in range(self.num_epochs):
            self._model.train()
            epoch_loss, n_correct, n_total = 0.0, 0, 0

            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labs = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self._model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(outputs.logits, labs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                n_correct += (preds == labs).sum().item()
                n_total += labs.size(0)

            train_acc = n_correct / max(n_total, 1)
            logger.info(
                "  Época %d/%d — loss: %.4f | train_acc: %.4f",
                epoch + 1, self.num_epochs, epoch_loss / len(loader), train_acc,
            )

            if val_texts is not None and val_labels is not None:
                val_preds = self.predict(val_texts)
                val_acc = float(np.mean(np.array(val_preds) == np.array(list(val_labels))))
                logger.info("  Val accuracy: %.4f", val_acc)

        return self

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def _forward_batched(self, texts: Sequence[str]) -> np.ndarray:
        """Inferencia por batches. Devuelve logits (n, 2)."""
        if self._model is None:
            raise RuntimeError(
                "Llama a fit() o load() antes de predict/predict_proba()."
            )
        try:
            import torch
        except ImportError:
            raise ImportError(_INSTALL_MSG)

        self._model.eval()
        all_logits = []

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start: start + self.batch_size])
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self._model(**enc)
                all_logits.append(outputs.logits.cpu().numpy())

        return np.concatenate(all_logits, axis=0)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Devuelve etiquetas predichas (0 = FAKE, 1 = REAL)."""
        return self._forward_batched(texts).argmax(axis=1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Devuelve P(REAL) ∈ [0, 1] para cada texto.

        Interfaz compatible con ``FakeNewsClassifier.predict_proba``.
        """
        logits = self._forward_batched(texts)
        # Softmax estable
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        return probs[:, 1]  # columna 1 = REAL

    def evaluate(self, texts: Sequence[str], labels: Sequence[int]) -> dict:
        """Devuelve classification_report y confusion_matrix."""
        from sklearn.metrics import classification_report, confusion_matrix
        preds = self.predict(texts)
        return {
            "report": classification_report(labels, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str | Path = config.TRANSFORMER_CLF_PATH) -> None:
        """Guarda el modelo fine-tuneado y el tokenizador en ``path/``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("Nada que guardar: entrena el modelo primero.")
        self._model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        meta = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
        }
        (path / "veritas_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("TransformerClassifier guardado en %s", path)

    @classmethod
    def load(cls, path: str | Path = config.TRANSFORMER_CLF_PATH) -> "TransformerClassifier":
        """Carga un modelo previamente guardado con ``save()``."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(_INSTALL_MSG)

        path = Path(path)
        meta_path = path / "veritas_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        inst = cls(
            model_name=meta.get("model_name", config.TRANSFORMER_MODEL_NAME),
            max_length=meta.get("max_length", config.TRANSFORMER_MAX_LENGTH),
            batch_size=meta.get("batch_size", config.TRANSFORMER_BATCH_SIZE),
        )
        inst._tokenizer = AutoTokenizer.from_pretrained(str(path))
        inst._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        inst._model = inst._model.to(inst.device)
        inst._model.eval()
        logger.info("TransformerClassifier cargado desde %s (device=%s)", path, inst.device)
        return inst
