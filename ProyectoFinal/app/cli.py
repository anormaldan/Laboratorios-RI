"""Interfaz de línea de comandos para búsqueda + verificación.

Ejemplo:
    python -m app.cli --query "vaccines cause autism" --model bm25 --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import SearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Búsqueda + verificación de noticias falsas",
    )
    parser.add_argument("--query", "-q", required=True, help="Consulta del usuario")
    parser.add_argument("--model", "-m", choices=["tfidf", "bm25", "semantic"], default="bm25")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=5000,
                        help="Tamaño del subset del corpus (None = completo)")
    parser.add_argument("--refit", action="store_true",
                        help="Vuelve a entrenar el clasificador aunque exista")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = None if args.sample <= 0 else args.sample

    print(f"\n[+] Construyendo pipeline (modelo={args.model}, sample={sample})…")
    pipe = SearchPipeline.build(model=args.model, sample_size=sample, force_refit=args.refit)

    print(f"\n[+] Buscando: {args.query!r}\n")
    results = pipe.search_and_verify(args.query, top_k=args.top_k)
    print(pipe.format(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
