"""Orquestador end-to-end: preprocesamiento → indexado → búsqueda + clasificación.

Uso típico:

    from src.pipeline import SearchPipeline
    pipe = SearchPipeline.build(model="bm25")
    results = pipe.search_and_verify("vaccines cause autism", top_k=5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config
from .classification import EvidenceVerifier, FakeNewsClassifier, VerificationResult
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


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float
    prob_real: float
    label_true: Optional[int] = None
    # Veredicto FEVER opcional (si el pipeline corre el EvidenceVerifier).
    fever_verdict: Optional[str] = None       # SUPPORTS | REFUTES | NEI
    fever_scores: Optional[dict] = None

    @property
    def verdict(self) -> str:
        return "REAL" if self.prob_real >= 0.5 else "FAKE"


@dataclass
class SearchPipeline:
    corpus: pd.DataFrame
    retriever: BaseRetriever
    classifier: FakeNewsClassifier
    verifier: Optional[EvidenceVerifier] = None  # FEVER-style opcional
    snippet_chars: int = 240
    history: List[dict] = field(default_factory=list)

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
    ) -> "SearchPipeline":
        """Construye el pipeline completo cargando el dataset real.

        Parameters
        ----------
        model : "tfidf" | "bm25" | "semantic"
            Recuperador base de la etapa 1.
        use_reranker : bool
            Si True, añade un CrossEncoderReranker como etapa 2.
        use_verifier : bool
            Si True, añade un EvidenceVerifier (FEVER) para etiquetar el veredicto
            por evidencia.
        """
        df = load_fake_real_news(sample_size=sample_size)
        df = preprocess_dataframe(df)

        retriever: BaseRetriever = _instantiate_retriever(model)
        retriever.fit(df["clean_text"].tolist())

        if use_reranker:
            retriever = RerankedRetriever(
                base=retriever,
                reranker=CrossEncoderReranker(),
                candidate_pool=100,
                documents=df["text"].tolist(),
            )

        classifier = FakeNewsClassifier()
        if config.CLASSIFIER_PATH.exists() and not force_refit:
            classifier = FakeNewsClassifier.load(config.CLASSIFIER_PATH)
        else:
            classifier.fit(df["text"].tolist(), df["label"].tolist())
            classifier.save(config.CLASSIFIER_PATH)

        verifier = EvidenceVerifier() if use_verifier else None

        return cls(corpus=df, retriever=retriever, classifier=classifier, verifier=verifier)

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------
    def search_and_verify(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[SearchResult]:
        indices, scores = self.retriever.search(query, top_k=top_k)
        docs = self.corpus.iloc[indices]
        probs = self.classifier.predict_proba(docs["text"].tolist())

        # Verificación FEVER opcional sobre el conjunto de evidencias recuperadas.
        fever_per_doc: List[Optional[VerificationResult]] = [None] * len(indices)
        if self.verifier is not None:
            evidences = docs["text"].tolist()
            for i, ev in enumerate(evidences):
                fever_per_doc[i] = self.verifier.verify(query, [ev])

        results: List[SearchResult] = []
        for rank, (idx, score, prob, fv) in enumerate(
            zip(indices, scores, probs, fever_per_doc), start=1
        ):
            row = self.corpus.iloc[int(idx)]
            results.append(SearchResult(
                rank=rank,
                doc_id=str(row["id"]),
                title=str(row.get("title", ""))[:120],
                snippet=str(row["text"])[:self.snippet_chars].strip() + "…",
                score=float(score),
                prob_real=float(prob),
                label_true=int(row["label"]) if "label" in row else None,
                fever_verdict=fv.verdict if fv else None,
                fever_scores=fv.scores if fv else None,
            ))

        self.history.append({"query": query, "results": [r.__dict__ for r in results]})
        return results

    def format(self, results: List[SearchResult]) -> str:
        lines = []
        for r in results:
            badge = "🟢 REAL" if r.verdict == "REAL" else "🔴 FAKE"
            lines.append(
                f"[{r.rank}] {badge}  P(real)={r.prob_real:.2f}  score={r.score:.3f}\n"
                f"    {r.title}\n    {r.snippet}\n"
            )
        return "\n".join(lines)


def _instantiate_retriever(model: str) -> BaseRetriever:
    model = model.lower()
    if model == "tfidf":
        return TfidfRetriever()
    if model == "bm25":
        return BM25Retriever()
    if model == "semantic":
        return SemanticRetriever()
    raise ValueError(f"Modelo desconocido: {model}. Usa tfidf|bm25|semantic.")


# ---------------------------------------------------------------------------
# Entry-point para evaluación comparativa rápida
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline RI + Fake News")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--model", choices=["tfidf", "bm25", "semantic"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()

    pipe = SearchPipeline.build(model=args.model, sample_size=args.sample)
    if args.query:
        print(pipe.format(pipe.search_and_verify(args.query, top_k=args.top_k)))
    else:
        print("Pipeline construido. Pasa --query para buscar.")


if __name__ == "__main__":
    main()
