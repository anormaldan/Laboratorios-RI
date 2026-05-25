"""Orquestador end-to-end: preprocesamiento → indexado → búsqueda + clasificación.

Mejoras integradas (mayo 2025)
------------------------------
* **TransformerClassifier** — Fine-tuning de ALBERT/DistilBERT como alternativa
  al clasificador TF-IDF+LogReg.  Inspirado en Azizah et al. (2023) (Paper 2):
  ALBERT alcanza 87.6 % accuracy vs 56 % de NB+TF-IDF (Sutradhar et al., Paper 1).

* **Extracción de evidencia a nivel de oración** — En lugar de pasar los primeros
  800 caracteres al verificador FEVER, se seleccionan las N frases del documento
  con mayor overlap léxico con el claim.  Mejora significativamente la precisión
  del módulo NLI.

* **Veredicto ensemble** — Combina P(real) del clasificador con la señal NLI de
  FEVER (SUPPORTS / REFUTES) en una puntuación ponderada para un veredicto más
  robusto y con menor varianza.

* **Consenso de evidencia** — Agrega los veredictos FEVER de todos los documentos
  recuperados en un objeto ``EvidenceConsensus`` que resume cuántos SUPPORTS vs
  REFUTES se encontraron.

Uso típico::

    from src.pipeline import SearchPipeline
    pipe = SearchPipeline.build(model="bm25", use_verifier=True)
    results, consensus = pipe.search_and_verify("vaccines cause autism", top_k=5)
    print(consensus)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .classification import EvidenceVerifier, FakeNewsClassifier, VerificationResult
from .classification.transformer_classifier import TransformerClassifier
from .data_loader import load_fake_real_news
from .preprocessing import preprocess_dataframe
from .retrieval import (
    BM25Retriever,
    BaseRetriever,
    CrossEncoderReranker,
    RerankedRetriever,
    SemanticRetriever,
    TfidfRetriever,
)


# ---------------------------------------------------------------------------
# Consenso de evidencia
# ---------------------------------------------------------------------------

@dataclass
class EvidenceConsensus:
    """Agrega los veredictos FEVER de todos los documentos recuperados.

    Permite responder: "¿cuántos de los X documentos encontrados REFUTAN
    el claim?" — un indicador de credibilidad complementario al P(real).
    """
    supports: int = 0
    refutes: int = 0
    nei: int = 0
    total: int = 0

    @property
    def dominant(self) -> str:
        """Veredicto dominante entre los documentos recuperados."""
        if self.total == 0:
            return "NEI"
        counts = {"SUPPORTS": self.supports, "REFUTES": self.refutes, "NEI": self.nei}
        return max(counts, key=counts.get)

    @property
    def support_ratio(self) -> float:
        return self.supports / max(self.total, 1)

    @property
    def refute_ratio(self) -> float:
        return self.refutes / max(self.total, 1)

    def __str__(self) -> str:
        return (
            f"Consenso ({self.total} docs): "
            f"SUPPORTS={self.supports} | REFUTES={self.refutes} | NEI={self.nei} "
            f"→ {self.dominant}"
        )


# ---------------------------------------------------------------------------
# Resultado de búsqueda
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float
    prob_real: float                        # P(real) del clasificador binario
    label_true: Optional[int] = None
    fever_verdict: Optional[str] = None    # SUPPORTS | REFUTES | NEI
    fever_scores: Optional[dict] = None
    combined_prob_real: Optional[float] = None  # ensemble clasificador + FEVER
    evidence_sentence: Optional[str] = None     # frase más relevante usada en NLI

    @property
    def verdict(self) -> str:
        """Veredicto final.

        Usa ``combined_prob_real`` cuando está disponible (ensemble activo),
        si no, cae al P(real) del clasificador binario.
        """
        score = (
            self.combined_prob_real
            if self.combined_prob_real is not None
            else self.prob_real
        )
        return "REAL" if score >= 0.5 else "FAKE"

    @property
    def confidence(self) -> float:
        """Distancia al umbral de decisión (0=mínima confianza, 0.5=máxima)."""
        score = (
            self.combined_prob_real
            if self.combined_prob_real is not None
            else self.prob_real
        )
        return abs(score - 0.5)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

@dataclass
class SearchPipeline:
    corpus: pd.DataFrame
    retriever: BaseRetriever
    classifier: FakeNewsClassifier | TransformerClassifier
    verifier: Optional[EvidenceVerifier] = None
    snippet_chars: int = 240
    history: List[dict] = field(default_factory=list)

    # Último consenso calculado (útil para el dashboard de Streamlit)
    last_consensus: Optional[EvidenceConsensus] = None

    # ------------------------------------------------------------------
    # Constructores
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model: str = "bm25",
        sample_size: Optional[int] = 10_000,
        force_refit: bool = False,
        use_reranker: bool = False,
        use_verifier: bool = False,
        use_transformer_clf: bool = False,
    ) -> "SearchPipeline":
        """Construye el pipeline completo cargando el dataset real.

        Parámetros
        ----------
        model : "tfidf" | "bm25" | "semantic"
            Recuperador base de la Etapa 1.
        use_reranker : bool
            Si True, añade CrossEncoderReranker como Etapa 2
            (Nogueira & Cho, 2019).
        use_verifier : bool
            Si True, añade EvidenceVerifier FEVER-style para etiquetar
            el veredicto por evidencia (Thorne et al., 2018).
        use_transformer_clf : bool
            Si True, sustituye el clasificador TF-IDF+LogReg por un
            TransformerClassifier fine-tuneado (ALBERT, Azizah et al., 2023).
            Requiere `pip install torch transformers`.
        """
        df = load_fake_real_news(sample_size=sample_size)
        df = preprocess_dataframe(df)

        # ── Recuperador ────────────────────────────────────────────────
        # Usa índice en disco si existe (generado por scripts/warmup.py)
        # para evitar re-entrenamiento en cada arranque de Streamlit.
        retriever: BaseRetriever = _load_or_fit_retriever(
            model, df["clean_text"].tolist(), force_refit
        )

        if use_reranker:
            retriever = RerankedRetriever(
                base=retriever,
                reranker=CrossEncoderReranker(),
                candidate_pool=100,
                documents=df["text"].tolist(),
            )

        # ── Clasificador ────────────────────────────────────────────────
        if use_transformer_clf:
            clf_path = config.TRANSFORMER_CLF_PATH
            if clf_path.exists() and not force_refit:
                classifier: FakeNewsClassifier | TransformerClassifier = (
                    TransformerClassifier.load(clf_path)
                )
            else:
                classifier = TransformerClassifier()
                classifier.fit(df["text"].tolist(), df["label"].tolist())
                classifier.save(clf_path)
        else:
            clf_path_tfidf = config.CLASSIFIER_PATH
            if clf_path_tfidf.exists() and not force_refit:
                classifier = FakeNewsClassifier.load(clf_path_tfidf)
            else:
                classifier = FakeNewsClassifier()
                classifier.fit(df["text"].tolist(), df["label"].tolist())
                classifier.save(clf_path_tfidf)

        # ── Verificador FEVER ──────────────────────────────────────────
        verifier = EvidenceVerifier() if use_verifier else None

        return cls(
            corpus=df,
            retriever=retriever,
            classifier=classifier,
            verifier=verifier,
        )

    # ------------------------------------------------------------------
    # Extracción de evidencia a nivel de oración
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_relevant_sentences(
        claim: str,
        text: str,
        n: int = config.FEVER_SENTENCE_N,
        max_chars: int = config.FEVER_SENTENCE_CHARS,
    ) -> str:
        """Selecciona las N frases más relevantes del texto para el claim.

        Usa overlap léxico de tokens (rápido, sin modelo adicional).
        Mejora la precisión del verificador NLI al focalizar la evidencia
        en los fragmentos que realmente hablan del claim, en lugar de
        pasar los primeros N caracteres arbitrarios.

        Inspirado en el pipeline de recuperación de evidencia de FEVER
        (Thorne et al., NAACL 2018).
        """
        # Segmentación por límites de oración
        sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:max_chars]
        if len(sentences) <= n:
            return " ".join(sentences)[:max_chars]

        # Scoring por overlap de tokens ≥ 3 caracteres
        claim_tokens = set(re.findall(r"\b\w{3,}\b", claim.lower()))
        if not claim_tokens:
            return " ".join(sentences[:n])[:max_chars]

        scored = []
        for sent in sentences:
            sent_tokens = set(re.findall(r"\b\w{3,}\b", sent.lower()))
            overlap = len(claim_tokens & sent_tokens) / len(claim_tokens)
            scored.append((overlap, sent))

        scored.sort(key=lambda x: -x[0])
        top_sentences = [s for _, s in scored[:n]]
        return " ".join(top_sentences)[:max_chars]

    # ------------------------------------------------------------------
    # Veredicto ensemble
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble_prob(
        prob_real: float,
        fever_result: Optional[VerificationResult],
        fever_weight: float = config.ENSEMBLE_FEVER_WEIGHT,
    ) -> float:
        """Combina P(real) del clasificador con la señal NLI de FEVER.

        Estrategia:
        - FEVER SUPPORTS → incrementa P(real) en proporción al score NLI
        - FEVER REFUTES  → decrementa P(real) en proporción al score NLI
        - FEVER NEI      → sin modificación

        El peso ``fever_weight`` (0.35 por defecto) controla cuánto influye
        FEVER respecto al clasificador base.  Es conservador para no sobre-
        corregir en casos donde el retriever devuelve documentos tangenciales.
        """
        if fever_result is None:
            return prob_real

        fever_signal = 0.0
        if fever_result.verdict == "SUPPORTS":
            fever_signal = +fever_result.scores.get("SUPPORTS", 0.0)
        elif fever_result.verdict == "REFUTES":
            fever_signal = -fever_result.scores.get("REFUTES", 0.0)
        # NEI: sin señal

        # Ajuste ponderado: señal FEVER ∈ [-1, +1] → δP ∈ [-0.35, +0.35]
        delta = fever_weight * fever_signal
        combined = float(np.clip(prob_real + delta, 0.0, 1.0))
        return combined

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def search_and_verify(
        self, query: str, top_k: int = config.DEFAULT_TOP_K
    ) -> Tuple[List[SearchResult], Optional[EvidenceConsensus]]:
        """Busca, clasifica y verifica una consulta.

        Devuelve
        --------
        results : List[SearchResult]
            Documentos rankeados con veredicto, P(real) y (si activo) FEVER.
        consensus : EvidenceConsensus | None
            Agregado de veredictos FEVER sobre todos los documentos devueltos.
            None si el verificador no está activo.
        """
        indices, scores = self.retriever.search(query, top_k=top_k)
        docs = self.corpus.iloc[indices]
        texts = docs["text"].tolist()
        probs = self.classifier.predict_proba(texts)

        # ── Verificación FEVER por documento ──────────────────────────
        fever_results: List[Optional[VerificationResult]] = [None] * len(indices)
        evidence_sentences: List[Optional[str]] = [None] * len(indices)

        if self.verifier is not None:
            for i, doc_text in enumerate(texts):
                # Extrae las frases más relevantes en lugar del texto crudo
                best_evidence = self._extract_relevant_sentences(query, doc_text)
                evidence_sentences[i] = best_evidence
                fever_results[i] = self.verifier.verify(query, [best_evidence])

        # ── Consenso global ───────────────────────────────────────────
        consensus: Optional[EvidenceConsensus] = None
        if self.verifier is not None:
            con = EvidenceConsensus(total=len(indices))
            for fv in fever_results:
                if fv is None:
                    con.nei += 1
                elif fv.verdict == "SUPPORTS":
                    con.supports += 1
                elif fv.verdict == "REFUTES":
                    con.refutes += 1
                else:
                    con.nei += 1
            consensus = con

        # ── Construcción de resultados ────────────────────────────────
        results: List[SearchResult] = []
        for rank, (idx, score, prob, fv, ev_sent) in enumerate(
            zip(indices, scores, probs, fever_results, evidence_sentences), start=1
        ):
            row = self.corpus.iloc[int(idx)]
            combined = self._ensemble_prob(float(prob), fv)

            results.append(SearchResult(
                rank=rank,
                doc_id=str(row["id"]),
                title=str(row.get("title", ""))[:120],
                snippet=str(row["text"])[: self.snippet_chars].strip() + "…",
                score=float(score),
                prob_real=float(prob),
                label_true=int(row["label"]) if "label" in row else None,
                fever_verdict=fv.verdict if fv else None,
                fever_scores=fv.scores if fv else None,
                combined_prob_real=combined if self.verifier is not None else None,
                evidence_sentence=ev_sent,
            ))

        self.last_consensus = consensus
        self.history.append({
            "query": query,
            "results": [r.__dict__ for r in results],
            "consensus": consensus.__dict__ if consensus else None,
        })
        return results, consensus

    def format(self, results: List[SearchResult]) -> str:
        lines = []
        for r in results:
            badge = "REAL" if r.verdict == "REAL" else "FAKE"
            ensemble_note = (
                f" [ensemble={r.combined_prob_real:.2f}]"
                if r.combined_prob_real is not None
                else ""
            )
            fever_note = f" | FEVER:{r.fever_verdict}" if r.fever_verdict else ""
            lines.append(
                f"[{r.rank}] {badge}  P(real)={r.prob_real:.2f}{ensemble_note}"
                f"  score={r.score:.3f}{fever_note}\n"
                f"    {r.title}\n    {r.snippet}\n"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instantiate_retriever(model: str) -> BaseRetriever:
    model = model.lower()
    if model == "tfidf":
        return TfidfRetriever()
    if model == "bm25":
        return BM25Retriever()
    if model == "semantic":
        return SemanticRetriever()
    raise ValueError(f"Modelo desconocido: {model}. Usa tfidf|bm25|semantic.")


def _load_or_fit_retriever(
    model: str,
    corpus: list[str],
    force_refit: bool = False,
) -> BaseRetriever:
    """Carga el índice desde disco si existe; si no, entrena y guarda.

    El caché en disco se genera con ``scripts/warmup.py``. Esto hace que
    el arranque de Streamlit sea prácticamente instantáneo después del warmup.
    """
    model = model.lower()

    disk_paths = {
        "bm25":     config.BM25_INDEX_PATH,
        "tfidf":    config.TFIDF_INDEX_PATH,
        "semantic": config.EMBEDDINGS_PATH,
    }
    loaders = {
        "bm25":  BM25Retriever.load,
        "tfidf": TfidfRetriever.load,
    }

    path = disk_paths.get(model)

    if model == "semantic":
        # El SemanticRetriever guarda embeddings como .npy + .meta
        meta_path = config.EMBEDDINGS_PATH.with_suffix(".meta")
        if not force_refit and path and path.exists() and meta_path.exists():
            retriever = SemanticRetriever.load(config.EMBEDDINGS_PATH)
            return retriever
        retriever = SemanticRetriever()
        retriever.fit(corpus)
        retriever.save(config.EMBEDDINGS_PATH)
        return retriever

    if path and not force_refit and path.exists():
        try:
            retriever = loaders[model](path)
            return retriever
        except Exception:
            pass  # índice corrupto → re-entrenar

    # Sin caché válido: entrenar y guardar
    retriever = _instantiate_retriever(model)
    retriever.fit(corpus)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        retriever.save(path)
    return retriever


# ---------------------------------------------------------------------------
# Entry-point para evaluación comparativa rápida
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline RI + Fake News — VERITAS")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--model", choices=["tfidf", "bm25", "semantic"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--verifier", action="store_true")
    parser.add_argument("--transformer", action="store_true")
    args = parser.parse_args()

    pipe = SearchPipeline.build(
        model=args.model,
        sample_size=args.sample,
        use_reranker=args.reranker,
        use_verifier=args.verifier,
        use_transformer_clf=args.transformer,
    )
    if args.query:
        results, consensus = pipe.search_and_verify(args.query, top_k=args.top_k)
        print(pipe.format(results))
        if consensus:
            print(consensus)
    else:
        print("Pipeline construido. Pasa --query para buscar.")


if __name__ == "__main__":
    main()
