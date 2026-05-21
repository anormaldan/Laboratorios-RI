# Mejoras sugeridas para VERITAS

Cada mejora incluye **(a)** justificación con paper, **(b)** esfuerzo estimado,
**(c)** impacto esperado y **(d)** dónde encajaría en el código.

Las **ordenadas por mejor relación esfuerzo / impacto** son ⭐ (recomendadas);
las demás son trabajo futuro.

---

## ⭐ Mejora 1 · Reranker cross-encoder (two-stage retrieval)

**Justificación:** Nogueira & Cho (2019) *Passage Re-ranking with BERT* demuestran
que un cross-encoder reordenando los top-100 de BM25 mejora MRR/NDCG en 20-30 %
respecto a BM25 solo. Es el patrón estándar en TREC desde 2019.

**Cómo:** usar `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parámetros). BM25
recupera 100 candidatos rápido → el cross-encoder los reordena con atención
query↔doc completa.

**Esfuerzo:** ⏱ 1 día. **Impacto:** 🚀🚀🚀 (alto en NDCG@5, NDCG@10).

**Dónde:** ✅ Ya implementado en `src/retrieval/reranker.py` (este commit).

```python
from src.retrieval import BM25Retriever, CrossEncoderReranker, RerankedRetriever
base = BM25Retriever().fit(corpus)
reranker = CrossEncoderReranker()
hybrid = RerankedRetriever(base, reranker, candidate_pool=100)
hybrid.search("vaccines cause autism", top_k=5)
```

---

## ⭐ Mejora 2 · Verificación FEVER-style con NLI (3 clases)

**Justificación:** Thorne et al. (2018) FEVER definen el pipeline de verificación
en 3 etiquetas: SUPPORTS / REFUTES / NOT ENOUGH INFO. Es muchísimo más
informativo que un binario fake/real y **justifica el nombre VERITAS**
(*Verification* and Retrieval).

**Cómo:** modelo de NLI pre-entrenado (`cross-encoder/nli-deberta-v3-base`).
Para cada documento recuperado, evaluar `(claim → snippet)` y producir
distribución sobre {entailment, contradiction, neutral} = {SUPPORTS, REFUTES, NEI}.

**Esfuerzo:** ⏱ 1 día. **Impacto:** 🚀🚀🚀 (alineamiento conceptual con el nombre,
diferenciador en exposición).

**Dónde:** ✅ Implementado en `src/classification/evidence_verifier.py`.

```python
from src.classification import EvidenceVerifier
verifier = EvidenceVerifier()
verifier.verify("vaccines cause autism", ["A 2023 meta-analysis with 1.2M children found no link…"])
# -> {"verdict": "REFUTES", "scores": {"SUPPORTS": 0.04, "REFUTES": 0.91, "NEI": 0.05}}
```

---

## ⭐ Mejora 3 · MRR + bootstrap CI en evaluación

**Justificación:** Manning et al. (2008) cap. 8 recomiendan reportar **MRR**
junto a P@K cuando hay 1-2 documentos relevantes por consulta. Además, agregar
intervalos de confianza por bootstrap (Sakai 2014) protege ante muestras
pequeñas de consultas.

**Esfuerzo:** ⏱ medio día. **Impacto:** 🚀🚀 (defensa estadística sólida).

**Dónde:** extender `src/evaluation/metrics.py` con `reciprocal_rank`,
`mean_reciprocal_rank` y un helper `bootstrap_ci(metric_fn, n=1000)`.

---

## Mejora 4 · Sub-corpus FEVER para evaluar verificación real

**Justificación:** Guo et al. (2022) *A survey on automated fact-checking* señalan
que evaluar verificación contra ground-truth real (no derivado de `subject`) es
crítico. FEVER ofrece 185k claims con evidencia anotada.

**Cómo:** descargar `train.jsonl` de FEVER, filtrar a 1-2k claims, usar VERITAS
sobre el corpus de Wikipedia que viene incluido.

**Esfuerzo:** ⏱ 2-3 días. **Impacto:** 🚀🚀 (rigor académico, métrica honesta).

---

## Mejora 5 · Dense retrieval con fine-tuning sobre el corpus

**Justificación:** Karpukhin et al. (2020) DPR muestran que entrenar el
dual-encoder con triplets (query, doc+, doc−) supera a BM25 en dominios
in-distribution. Para VERITAS, usar `MultipleNegativesRankingLoss` de
sentence-transformers.

**Cómo:** crear pares `(titular, cuerpo)` del mismo artículo como positivos y
muestreo aleatorio como negativos.

**Esfuerzo:** ⏱ 3-5 días. **Impacto:** 🚀🚀 (mejora moderada en datos in-domain).

---

## Mejora 6 · Generación de explicaciones del veredicto

**Justificación:** Atanasova et al. (2020) *Generating fact checking explanations*
muestran que producir una justificación textual aumenta la confianza del usuario.

**Cómo:** highlighting de las oraciones del documento con mayor `entailment_prob`
hacia el claim (atención del cross-encoder). No requiere LLM.

**Esfuerzo:** ⏱ medio día. **Impacto:** 🚀 (mejora UX y nota de "claridad").

---

## Mejora 7 · Multilingüe (español)

**Justificación:** El corpus actual es 100 % inglés. Para que VERITAS sea
demostrable con noticias en español:
- Reemplazar `all-MiniLM-L6-v2` por `paraphrase-multilingual-MiniLM-L12-v2`
  (Reimers & Gurevych 2020).
- Reemplazar stopwords NLTK por `stopwords-iso` español.
- Usar el dataset **Spanish Fake News Corpus** (Posadas-Durán et al. 2019).

**Esfuerzo:** ⏱ 2-3 días. **Impacto:** 🚀🚀 (relevante regionalmente).

---

## Mejora 8 · Indexado con FAISS para corpus >100k

**Justificación:** Johnson, Douze, & Jégou (2019) FAISS es el estándar de la
industria para nearest-neighbor sobre embeddings. Con IVF/HNSW se baja la
búsqueda semántica de O(n) a O(log n).

**Cómo:** wrapper en `src/retrieval/semantic_retriever.py` cuando
`embeddings.shape[0] > 50_000`.

**Esfuerzo:** ⏱ 1 día. **Impacto:** 🚀 (solo si escalan el corpus, no antes).

---

## Mejora 9 · Detección de sesgo y data leakage

**Justificación:** El Fake/Real News dataset de Kaggle tiene un **data leakage
conocido**: los artículos REAL provienen casi todos de Reuters (firma identificable
en el texto), por lo que el clasificador aprende a detectar "Reuters" en vez de
contenido real. Bozarth & Budak (2020) lo documentan.

**Cómo:**
1. Eliminar la firma `(Reuters)` y nombres de agencia en el preprocesamiento.
2. Reportar métricas con y sin el limpiador para mostrar conciencia del problema.

**Esfuerzo:** ⏱ 2 horas. **Impacto:** 🚀🚀 (honestidad académica, evita ser
"cazados" en preguntas).

```python
LEAK_PATTERNS = re.compile(r"\(reuters\)|—\s*reuters|reuters\s*-", re.IGNORECASE)
text = LEAK_PATTERNS.sub("", text)
```

---

## Mejora 10 · App con histórico y export

**Justificación:** Para defensa oral, poder mostrar 3-4 queries demo
pre-guardadas evita silencios incómodos. Streamlit lo permite con
`st.session_state`.

**Esfuerzo:** ⏱ 2 horas. **Impacto:** 🚀 (calidad de demo).

---

## Priorización recomendada (tiempo limitado a 4 semanas extra)

| Orden | Mejora | Tiempo | Por qué primero |
|---|---|---|---|
| 1 | #9 Data leakage Reuters | 2 h | Bug crítico documentado en la literatura |
| 2 | #1 Reranker cross-encoder | 1 d | Mayor impacto en NDCG/MAP |
| 3 | #2 Verificación FEVER-style | 1 d | Justifica el nombre del sistema |
| 4 | #3 MRR + bootstrap CI | 0.5 d | Rigor en la evaluación |
| 5 | #6 Highlight de evidencia | 0.5 d | Mejora claridad para exposición |
| 6 | #4 Sub-corpus FEVER | 2-3 d | Evaluación honesta de verificación |

Las mejoras 5, 7, 8 quedan como **trabajo futuro** en el reporte.
