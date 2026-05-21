"""Carga de datasets crudos a DataFrames unificados.

Convención: el DataFrame devuelto tiene siempre las columnas
``id, title, text, subject, label`` donde ``label = 1`` significa REAL y
``label = 0`` significa FAKE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def load_fake_real_news(
    raw_dir: Optional[Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = config.SEED,
) -> pd.DataFrame:
    """Carga el Fake and Real News Dataset de Kaggle.

    Espera ``Fake.csv`` y ``True.csv`` en ``raw_dir``.

    Parameters
    ----------
    sample_size : int, optional
        Si se indica, devuelve una muestra estratificada por clase de ese tamaño.
    """
    raw_dir = Path(raw_dir or config.RAW_DIR)
    fake_path = raw_dir / "Fake.csv"
    real_path = raw_dir / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            f"No se encontraron Fake.csv / True.csv en {raw_dir}. "
            "Consulta data/raw/README.md para descargarlos."
        )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real], ignore_index=True)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["id"] = df.index.astype(str)

    for col in ("title", "subject"):
        if col not in df.columns:
            df[col] = ""

    df = df[["id", "title", "text", "subject", "label"]]

    if sample_size is not None and sample_size < len(df):
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), sample_size // 2), random_state=random_state))
            .reset_index(drop=True)
        )

    return df


def load_liar(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Carga el LIAR Dataset. Mapea 6 clases a binario REAL/FAKE."""
    raw_dir = Path(raw_dir or config.RAW_DIR) / "liar"
    columns = [
        "id", "label_str", "statement", "subject", "speaker", "job",
        "state", "party", "barely_true", "false", "half_true",
        "mostly_true", "pants_on_fire", "context",
    ]

    frames = []
    for split in ("train", "valid", "test"):
        path = raw_dir / f"{split}.tsv"
        if not path.exists():
            continue
        part = pd.read_csv(path, sep="\t", header=None, names=columns)
        part["split"] = split
        frames.append(part)

    if not frames:
        raise FileNotFoundError(f"No se encontró LIAR en {raw_dir}.")

    df = pd.concat(frames, ignore_index=True)

    label_map = {
        "true": 1, "mostly-true": 1, "half-true": 1,
        "barely-true": 0, "false": 0, "pants-fire": 0,
    }
    df["label"] = df["label_str"].map(label_map)
    df = df.rename(columns={"statement": "text"})
    df["title"] = ""
    return df[["id", "title", "text", "subject", "label"]]


def train_val_test_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.SEED,
):
    """Split estratificado 70/15/15 por defecto."""
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=test_size + val_size,
        stratify=df["label"], random_state=random_state,
    )
    rel = test_size / (test_size + val_size)
    val, test = train_test_split(
        temp, test_size=rel, stratify=temp["label"], random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    df = load_fake_real_news(sample_size=2000)
    print(df.head())
    print(f"\nTotal: {len(df)} | Balance: {df['label'].value_counts().to_dict()}")
