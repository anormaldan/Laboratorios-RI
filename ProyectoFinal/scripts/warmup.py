"""Script de preparación previa a la presentación.

Ejecutar UNA VEZ antes del demo (la noche anterior o con anticipación):

    python scripts/warmup.py

Qué hace
--------
1. Construye y guarda en disco los índices BM25 y TF-IDF
2. Entrena y guarda el clasificador LogReg si no existe
3. Pre-descarga los modelos HuggingFace (reranker + FEVER NLI)
   al caché local para que Streamlit no los descargue en plena demo
4. Reporta el tiempo de cada paso

Después de esto, el arranque de Streamlit es prácticamente instantáneo:
el pipeline carga todo desde disco en lugar de reentrenar.
"""
import sys
import time
from pathlib import Path

# Aseguramos que src/ esté en el path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.classification import FakeNewsClassifier
from src.data_loader import load_fake_real_news
from src.preprocessing import preprocess_dataframe
from src.retrieval import BM25Retriever, TfidfRetriever


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def tick(label: str, t0: float) -> None:
    elapsed = time.time() - t0
    print(f"  [OK] {label:45s} {elapsed:5.1f}s")


def main() -> None:
    print("\n⚖️  VERITAS — Script de warmup para demo / presentación")
    print("   Ejecuta esto una vez antes de presentar.\n")

    total_start = time.time()

    # ── 1. Cargar datos ────────────────────────────────────────────────
    banner("Paso 1/5 · Carga de datos")
    t = time.time()
    SAMPLE = 5000
    df = load_fake_real_news(sample_size=SAMPLE)
    df = preprocess_dataframe(df)
    print(f"  Corpus: {len(df)} documentos  |  "
          f"FAKE: {(df['label']==0).sum()}  REAL: {(df['label']==1).sum()}")
    tick("Carga + preprocesamiento", t)

    corpus = df["clean_text"].tolist()
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # ── 2. Índice BM25 ─────────────────────────────────────────────────
    banner("Paso 2/5 · Índice BM25")
    bm25_path = config.BM25_INDEX_PATH
    if bm25_path.exists():
        print(f"  Ya existe: {bm25_path}  — omitiendo.")
    else:
        t = time.time()
        bm25 = BM25Retriever().fit(corpus)
        bm25.save(bm25_path)
        tick(f"BM25 fit + guardado en {bm25_path.name}", t)

    # ── 3. Índice TF-IDF ───────────────────────────────────────────────
    banner("Paso 3/5 · Índice TF-IDF")
    tfidf_path = config.TFIDF_INDEX_PATH
    if tfidf_path.exists():
        print(f"  Ya existe: {tfidf_path}  — omitiendo.")
    else:
        t = time.time()
        tfidf = TfidfRetriever().fit(corpus)
        tfidf.save(tfidf_path)
        tick(f"TF-IDF fit + guardado en {tfidf_path.name}", t)

    # ── 4. Clasificador LogReg ─────────────────────────────────────────
    banner("Paso 4/5 · Clasificador TF-IDF + LogReg")
    clf_path = config.CLASSIFIER_PATH
    if clf_path.exists():
        print(f"  Ya existe: {clf_path}  — omitiendo.")
    else:
        t = time.time()
        clf = FakeNewsClassifier()
        clf.fit(texts, labels)
        clf.save(clf_path)
        tick(f"LogReg entrenado + guardado en {clf_path.name}", t)

    # ── 5. Pre-descarga de modelos HuggingFace ─────────────────────────
    banner("Paso 5/5 · Pre-descarga de modelos HuggingFace")
    print("  (Se descargan solo si no están en caché local)\n")

    models = [
        ("Reranker  — ms-marco-MiniLM-L-6-v2",
         "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("FEVER NLI — nli-deberta-v3-base",
         "cross-encoder/nli-deberta-v3-base"),
        ("Semantic  — all-MiniLM-L6-v2",
         "sentence-transformers/all-MiniLM-L6-v2"),
    ]

    for label, model_id in models:
        t = time.time()
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            if "nli" in model_id or "marco" in model_id:
                CrossEncoder(model_id)
            else:
                SentenceTransformer(model_id)
            tick(label, t)
        except Exception as e:
            print(f"  [WARN] {label}: {e}")

    # ── Resumen ────────────────────────────────────────────────────────
    total = time.time() - total_start
    banner("Warmup completado")
    print(f"  Tiempo total: {total:.1f}s\n")

    print("  Archivos generados:")
    for path in [config.BM25_INDEX_PATH, config.TFIDF_INDEX_PATH, config.CLASSIFIER_PATH]:
        exists = "OK" if path.exists() else "FALTA"
        size = f"{path.stat().st_size/1024:.0f} KB" if path.exists() else "-"
        print(f"    [{exists}] {path.name:35s} {size}")

    print()
    print("  Ahora inicia Streamlit:")
    print("  > streamlit run app/streamlit_app.py")
    print()
    print("  El pipeline cargará todo desde disco — sin esperas durante el demo.")


if __name__ == "__main__":
    main()
