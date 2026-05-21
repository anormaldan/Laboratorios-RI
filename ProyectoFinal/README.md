# VERITAS
### *Verification and Retrieval Intelligent Truth Analysis System*

Proyecto final del curso **Recuperación de Información**.

| | |
|---|---|
| **Jonathan Tiro Cuanenemi** | 202268865 |
| **Alejandro Dante Nava Campos** | 202245606 |
| **Andrea Ortega Pérez** | 202246227 |

## Resumen

VERITAS es un sistema híbrido que combina técnicas clásicas de RI (TF-IDF, BM25),
representación semántica con embeddings y una capa de verificación inspirada en
FEVER para distinguir noticias reales de falsas. El usuario ingresa una afirmación
o titular y el sistema:

1. **Recupera** noticias relevantes ordenadas por similitud (TF-IDF / BM25 / dense).
2. **Reordena** (opcional) con un cross-encoder para mejorar el top-K.
3. **Verifica** la confiabilidad del contenido recuperado:
   - Modo binario: P(real) vs P(fake) con clasificador entrenado.
   - Modo evidencia: SUPPORTS / REFUTES / NEI usando NLI sobre los documentos top.
4. Permite **comparar** el desempeño de los enfoques (léxico, probabilístico, semántico, re-ranqueado).

## Estructura del proyecto

```
ProyectoFinal/
├── data/
│   ├── raw/            # CSVs originales (Fake.csv, True.csv, liar)
│   ├── processed/      # Corpus limpio (parquet/csv)
│   └── queries/        # Consultas de evaluación
├── notebooks/          # Jupyter para exploración y experimentos
├── src/                # Código fuente reutilizable
│   ├── retrieval/      # TF-IDF, BM25, embeddings, cross-encoder reranker
│   ├── classification/ # Clasificador binario + verificador FEVER-style
│   └── evaluation/     # Métricas
├── app/                # CLI + Streamlit demo
├── tests/              # Pruebas unitarias
├── docs/               # Plan, arquitectura, papers, mejoras
└── results/            # Figuras y reportes generados
```

## Instalación rápida

```powershell
# 1. Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar recursos NLTK (una sola vez)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 4. Colocar los datasets descargados en data/raw/
#    Fake.csv y True.csv desde Kaggle: clmentbisaillon/fake-and-real-news-dataset
```

## Uso

### CLI (búsqueda + verificación)
```powershell
python -m app.cli --query "vaccines cause autism" --top-k 5 --model bm25
```

### Streamlit (demo visual)
```powershell
streamlit run app/streamlit_app.py
```

### Evaluación comparativa
```powershell
python -m src.pipeline --eval --models tfidf bm25 semantic
```

## Documentación

- [Plan de acción](docs/plan_accion.md) — 8 semanas, 6 fases
- [Arquitectura del sistema](docs/arquitectura.md)
- [Rúbrica vs. propuesta](docs/rubrica_checklist.md)
- [Papers de referencia](docs/papers_referencia.md) — literatura científica usada
- [Mejoras sugeridas](docs/mejoras_sugeridas.md) — upgrades opcionales con citas

## Datasets

| Dataset | Uso | Fuente |
|---|---|---|
| Fake and Real News | Principal — corpus e índice | Kaggle: `clmentbisaillon/fake-and-real-news-dataset` |
| LIAR | Secundario — afirmaciones cortas | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |
| FEVER (opcional) | Verificación con evidencia | https://fever.ai/dataset/fever.html |
| FakeNewsNet | Opcional — extensión | https://github.com/KaiDMML/FakeNewsNet |

## Equipo

- **Jonathan Tiro Cuanenemi** · 202268865
- **Alejandro Dante Nava Campos** · 202245606
- **Andrea Ortega Pérez** · 202246227

> *Veritas — Verdad, en latín. Diosa romana de la verdad, hija de Saturno.*
