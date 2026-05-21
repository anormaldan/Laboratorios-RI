# Arquitectura de VERITAS

> *Verification and Retrieval Intelligent Truth Analysis System*

## Vista de alto nivel

```
                   ┌──────────────────────────────────┐
                   │         Usuario / Demo           │
                   │  (Streamlit · CLI · Notebook)    │
                   └────────────────┬─────────────────┘
                                    │ query
                                    ▼
                   ┌──────────────────────────────────┐
                   │           Pipeline               │
                   │   (src/pipeline.py · orquesta)   │
                   └──┬─────────────┬────────────┬────┘
                      │             │            │
                      ▼             ▼            ▼
              ┌──────────────┐ ┌──────────┐ ┌──────────┐
              │Preprocesamie-│ │Recupera- │ │Clasifica-│
              │nto           │ │ción      │ │dor       │
              └──────┬───────┘ └────┬─────┘ └────┬─────┘
                     │              │            │
                     │     ┌────────┼────────┐   │
                     │     ▼        ▼        ▼   │
                     │   TF-IDF   BM25    Semánt.│
                     │     │        │        │   │
                     └─────┴────────┴────────┴───┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │  Resultado: top-K docs │
                       │  + score recuperación  │
                       │  + prob_fake           │
                       └────────────────────────┘
```

## Componentes

### `src/config.py`
Constantes centrales: rutas, semilla, parámetros de modelos. Permite cambiar
hiperparámetros sin tocar el código.

### `src/data_loader.py`
Carga datasets crudos:
- `load_fake_real_news()` — combina `Fake.csv` + `True.csv` con etiqueta `label`.
- `load_liar()` — TSV con 6 clases que se mapean a binario.

### `src/preprocessing.py`
Función `clean_text(text)` y `preprocess_dataframe(df)`. Pipeline de NLTK:
lowercase → remove URLs → tokenize → remove stopwords → lemmatize.

### `src/retrieval/`
- `base.py` — clase abstracta `BaseRetriever` con `fit(corpus)` y `search(query, top_k)`.
- `tfidf_retriever.py` — `TfidfVectorizer` + `cosine_similarity`.
- `bm25_retriever.py` — `rank_bm25.BM25Okapi`.
- `semantic_retriever.py` — `SentenceTransformer` con cache de embeddings.

Todos los recuperadores exponen el mismo método `search` → tupla `(indices, scores)`.

### `src/classification/fake_news_classifier.py`
Wrapper sobre `LogisticRegression` (o `MultinomialNB`) entrenado con vectores
TF-IDF. Métodos `fit`, `predict_proba`, `save`, `load`.

### `src/evaluation/metrics.py`
Implementación clara (didáctica) de:
- `precision_at_k`, `recall_at_k`
- `average_precision`, `mean_average_precision`
- `dcg_at_k`, `ndcg_at_k`

### `src/pipeline.py`
Función `search_and_verify(query, retriever, classifier, top_k)` que devuelve
una lista de diccionarios listos para mostrar al usuario.

### `app/`
- `cli.py` — argparse-driven, imprime resultados a consola.
- `streamlit_app.py` — UI web con caja de búsqueda, selector de modelo, top-K, badge de confiabilidad.

## Flujo de una consulta (caso típico)

1. Usuario escribe `"vaccines cause autism in children"`.
2. `preprocessing.clean_text` la normaliza.
3. Recuperador (TF-IDF/BM25/semántico) devuelve top-5 índices del corpus.
4. Para cada doc, `classifier.predict_proba` estima P(fake).
5. Pipeline empaqueta `{title, snippet, score, prob_fake, label_real}`.
6. App muestra resultados con código de color (verde/rojo).

## Decisiones técnicas

| Decisión | Justificación |
|---|---|
| `MiniLM-L6-v2` sobre BERT-base | 5× más rápido, 90% del desempeño en STS |
| `joblib` para persistencia | Compatible con sklearn, comprime numpy arrays |
| `parquet` para corpus | 10× más rápido que CSV, tipa columnas |
| Logistic Regression sobre TF-IDF | Baseline fuerte, explica coeficientes |
| Streamlit sobre Flask | Cero boilerplate, hot-reload, ideal para demos académicas |

## Reproducibilidad

- Semilla global en `src/config.py` (`SEED = 42`).
- Todos los splits con `train_test_split(..., random_state=SEED, stratify=y)`.
- Versiones congeladas en `requirements.txt`.
