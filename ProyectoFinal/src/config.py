"""Configuración central del proyecto.

Edita aquí los hiperparámetros y rutas en lugar de hardcodearlos en otros módulos.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUERIES_DIR = DATA_DIR / "queries"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

CORPUS_PATH = PROCESSED_DIR / "corpus.parquet"
TFIDF_INDEX_PATH = PROCESSED_DIR / "tfidf_index.joblib"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.joblib"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
CLASSIFIER_PATH = PROCESSED_DIR / "classifier.joblib"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
LANGUAGE = "english"
MIN_TOKEN_LEN = 2
REMOVE_DIGITS = False
USE_LEMMATIZATION = True

# ---------------------------------------------------------------------------
# Modelos de recuperación
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

BM25_PARAMS = {
    "k1": 1.5,
    "b": 0.75,
}

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------
CLASSIFIER_TYPE = "logreg"  # "logreg" | "nb"
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ---------------------------------------------------------------------------
# Clasificador Transformer (ALBERT / DistilBERT)
# Motivación: Azizah et al. (2023) — ALBERT alcanza 87.6 % accuracy y
# 86.9 % F1 en detección de fake news, superando a BERT y RoBERTa con
# menor costo computacional (174.5 s/época).
# ---------------------------------------------------------------------------
TRANSFORMER_MODEL_NAME = "albert-base-v2"   # Paper 2 winner; también soporta
                                             # "distilbert-base-uncased" (más rápido)
TRANSFORMER_CLF_PATH = PROCESSED_DIR / "transformer_clf"
TRANSFORMER_MAX_LENGTH = 256   # titular + primer párrafo en ~256 tokens
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3         # 2-3 épocas son suficientes en datos in-domain
TRANSFORMER_LR = 2e-5          # estándar para fine-tuning BERT-like

# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
HYBRID_ALPHA = 0.5  # combinación BM25 + semántico

# ---------------------------------------------------------------------------
# Verificación FEVER
# ---------------------------------------------------------------------------
FEVER_SENTENCE_N = 3        # nº de oraciones a extraer por evidencia
FEVER_SENTENCE_CHARS = 700  # longitud máxima de la evidencia recortada
ENSEMBLE_FEVER_WEIGHT = 0.35  # peso del señal NLI en el veredicto combinado
