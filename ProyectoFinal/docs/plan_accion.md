# Plan de Acción — VERITAS

> **VERITAS** · *Verification and Retrieval Intelligent Truth Analysis System*
> Equipo: Jonathan Tiro · Alejandro Nava · Andrea Ortega

Plan dividido en **6 fases** sobre **8 semanas**. Cada fase produce un entregable
verificable y se alinea con un criterio de la rúbrica.

## Resumen de fases

| Fase | Semana | Entregable | Criterio rúbrica |
|---|---|---|---|
| 1. Setup y datos | 1 | Repo + dataset descargado + EDA | Dataset, Viabilidad |
| 2. Preprocesamiento | 2 | Corpus limpio en `data/processed/` | Metodología |
| 3. Recuperación clásica | 3-4 | TF-IDF + BM25 funcionando | Técnicas del curso |
| 4. Recuperación semántica | 5 | Embeddings + búsqueda híbrida | Originalidad |
| 5. Verificación + Evaluación | 6 | Clasificador + métricas comparativas | Comparación |
| 6. Demo + Reporte | 7-8 | App Streamlit + presentación | Claridad |

---

## Fase 1 — Setup y exploración (semana 1)

**Objetivo:** dejar el entorno listo y comprender los datos.

Tareas:
- [ ] Crear entorno virtual e instalar `requirements.txt`.
- [ ] Descargar **Fake and Real News Dataset** desde Kaggle a `data/raw/`.
- [ ] Ejecutar `notebooks/01_exploracion_datos.ipynb`.
- [ ] Documentar: número de documentos, balance de clases, longitud media,
      distribución por tema (`subject`), distribución temporal.

Entregable: figura `results/figures/eda_resumen.png` + sección en el reporte.

---

## Fase 2 — Preprocesamiento (semana 2)

**Objetivo:** producir un corpus limpio y reproducible.

Tareas:
- [ ] Lowercase, eliminación de URLs, símbolos, dígitos opcionales.
- [ ] Eliminación de stopwords (NLTK English).
- [ ] Lematización (WordNet) — opcional pero recomendado.
- [ ] Guardar `data/processed/corpus.parquet` con columnas `id, title, text, clean_text, label`.

Entregable: pipeline `src/preprocessing.py` y notebook `02_preprocesamiento.ipynb`.

---

## Fase 3 — Recuperación clásica (semanas 3-4)

**Objetivo:** implementar los dos baselines exigidos por la rúbrica.

Tareas:
- [ ] **TF-IDF**: `TfidfVectorizer` + similitud coseno (`src/retrieval/tfidf_retriever.py`).
- [ ] **BM25**: `rank_bm25.BM25Okapi` (`src/retrieval/bm25_retriever.py`).
- [ ] Clase base `BaseRetriever` con métodos `fit`, `search(query, top_k)`.
- [ ] Serializar índices con `joblib` en `data/processed/`.

Entregable: `notebooks/03_modelos_recuperacion.ipynb` con ejemplos de consulta.

---

## Fase 4 — Recuperación semántica (semana 5)

**Objetivo:** añadir representación semántica con embeddings.

Tareas:
- [ ] Usar `sentence-transformers/all-MiniLM-L6-v2` (ligero, ~80MB).
- [ ] Calcular embeddings del corpus una vez y persistirlos en `.npy`.
- [ ] Búsqueda por similitud coseno en GPU/CPU.
- [ ] (Opcional) Búsqueda híbrida: combinación lineal `α·BM25 + (1-α)·semantic`.

Entregable: `notebooks/04_modelos_semanticos.ipynb`.

---

## Fase 5 — Verificación y evaluación (semana 6)

**Objetivo:** clasificador de fake news + métricas comparativas.

Tareas:
- [ ] Entrenar Logistic Regression sobre TF-IDF (rápido) — `src/classification/`.
- [ ] (Comparar) Naive Bayes multinomial como segundo clasificador.
- [ ] Reportar Accuracy, Precision, Recall, F1, matriz de confusión.
- [ ] Evaluar **recuperación** con `Precision@K`, `Recall@K`, `MAP`, `NDCG@K` usando
      consultas con `ground truth` simulado a partir de `subject`/cluster.
- [ ] Generar tabla comparativa `results/reports/comparativa.csv`.

Entregable: `notebooks/06_evaluacion_comparativa.ipynb` + figuras.

---

## Fase 6 — Demo y reporte final (semanas 7-8)

**Objetivo:** presentación lista para defender.

Tareas:
- [ ] App Streamlit con caja de búsqueda, top-K resultados, badge de confiabilidad.
- [ ] CLI para reproducir resultados (`app/cli.py`).
- [ ] Documento final (`docs/reporte_final.md`) — extiende la propuesta.
- [ ] Presentación con flujo: problema → datos → métodos → resultados → demo.

Entregable: app funcional + PDF/PPT.

---

## Cronograma visual

```
Sem:  1   2   3   4   5   6   7   8
F1:  ██
F2:      ██
F3:          ████████
F4:                  ██
F5:                      ██
F6:                          ████████
```

## Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| Dataset muy grande (>40k docs) | Subsample estratificado de 10k para iterar |
| Embeddings lentos en CPU | Modelo MiniLM + batch size 64 + cache en disco |
| Ground truth de relevancia ausente | Usar `subject` como proxy o anotar 30 queries a mano |
| Sobreajuste del clasificador | Train/val/test 70/15/15 + cross-validation |
