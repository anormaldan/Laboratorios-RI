"""Pruebas del módulo de preprocesamiento."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import clean_text, preprocess_dataframe, tokenize


def test_clean_text_lowercases_and_strips_urls():
    out = clean_text("Visit https://x.com NOW for FREE pizza!!!")
    assert "http" not in out
    assert out == out.lower()
    assert "free" in out
    assert "pizza" in out


def test_clean_text_removes_stopwords():
    out = clean_text("the quick brown fox")
    assert "the" not in out.split()


def test_clean_text_handles_empty_and_nonstring():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_clean_text_is_idempotent():
    sample = "Breaking: vaccines cause AUTISM in children https://fake.io"
    once = clean_text(sample)
    twice = clean_text(once)
    assert once == twice


def test_tokenize_returns_list_of_strings():
    tokens = tokenize("Hello, world! Welcome to fake news class.")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert all(t == t.lower() for t in tokens)


def test_preprocess_dataframe_adds_clean_column():
    df = pd.DataFrame({
        "title": ["Hello world", "Fake news alert"],
        "text": ["The quick brown fox", "Vaccines are dangerous"],
    })
    out = preprocess_dataframe(df)
    assert "clean_text" in out.columns
    assert len(out) == 2
    assert all(out["clean_text"].str.len() > 0)
