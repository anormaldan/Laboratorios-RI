"""Pipeline de limpieza y normalización de texto.

Diseñado para ser idempotente: aplicar dos veces produce el mismo resultado.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

import pandas as pd

from . import config

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-záéíóúñü\s]" if False else r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")

# Mejora #9 — Data leakage del Fake/Real News Dataset:
# las noticias REAL casi todas vienen de Reuters y dejan una firma
# detectable. Si no se elimina, el clasificador aprende "Reuters" en lugar
# del estilo del texto (Bozarth & Budak, 2020).
_LEAK_PATTERNS = re.compile(
    r"\(reuters\)|—\s*reuters|reuters\.com|reuters\s+\-",
    re.IGNORECASE,
)


def remove_known_leaks(text: str) -> str:
    """Quita patrones identificadores del Fake/Real News Dataset.

    Aplicar ANTES de `clean_text` cuando se entrene el clasificador.
    """
    if not isinstance(text, str):
        return ""
    return _LEAK_PATTERNS.sub(" ", text)


@lru_cache(maxsize=1)
def _get_stopwords() -> set[str]:
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words(config.LANGUAGE))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words(config.LANGUAGE))


@lru_cache(maxsize=1)
def _get_lemmatizer():
    import nltk
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
        return lemmatizer
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer()


def clean_text(
    text: str,
    remove_digits: bool = config.REMOVE_DIGITS,
    use_lemmatization: bool = config.USE_LEMMATIZATION,
    min_token_len: int = config.MIN_TOKEN_LEN,
) -> str:
    """Aplica limpieza ligera y devuelve un string normalizado.

    Pasos: lowercase → quita URLs → quita no-alfabéticos → opcional dígitos
    → tokeniza por whitespace → quita stopwords → lematiza → filtra por longitud.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = text.lower()
    text = _LEAK_PATTERNS.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    if remove_digits:
        text = _DIGIT_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    stopwords = _get_stopwords()
    tokens = [t for t in text.split() if t not in stopwords and len(t) >= min_token_len]

    if use_lemmatization:
        lemmatizer = _get_lemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """Tokenización ligera para BM25 (no requiere NLTK)."""
    return clean_text(text).split()


def preprocess_dataframe(
    df: pd.DataFrame,
    text_columns: Iterable[str] = ("title", "text"),
    output_column: str = "clean_text",
) -> pd.DataFrame:
    """Aplica `clean_text` sobre la concatenación de varias columnas."""
    df = df.copy()
    df[output_column] = (
        df[list(text_columns)].fillna("").agg(" ".join, axis=1).map(clean_text)
    )
    df = df[df[output_column].str.len() > 0].reset_index(drop=True)
    return df


if __name__ == "__main__":
    sample = "Breaking: According to https://fake.com vaccines cause AUTISM!!! (study)."
    print("Original:", sample)
    print("Limpio:  ", clean_text(sample))
