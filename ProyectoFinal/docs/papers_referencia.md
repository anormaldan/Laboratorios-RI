# Papers de referencia para VERITAS

Literatura científica que sustenta cada componente del sistema. Útil para la
sección de **estado del arte** del reporte final y para defender preguntas
durante la exposición.

---

## 1. Fundamentos de RI clásica

### TF-IDF y modelo vectorial
- **Salton, G., Wong, A., & Yang, C. S. (1975).** *A vector space model for automatic indexing.* Communications of the ACM, 18(11), 613–620.
  → Paper fundacional del modelo vectorial y la similitud coseno.
- **Sparck Jones, K. (1972).** *A statistical interpretation of term specificity and its application in retrieval.* Journal of Documentation, 28(1), 11–21.
  → Origen del IDF.

### BM25
- **Robertson, S., & Zaragoza, H. (2009).** *The probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389.
  → Derivación completa de BM25 y sus parámetros `k1`, `b`. Cita obligatoria al justificar por qué BM25 supera a TF-IDF.

---

## 2. Detección de noticias falsas — marco general

- **Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017).** *Fake news detection on social media: A data mining perspective.* ACM SIGKDD Explorations Newsletter, 19(1), 22–36.
  → **Survey fundacional.** Define la taxonomía content-based vs. social-context-based que VERITAS implementa parcialmente.

- **Zhou, X., & Zafarani, R. (2020).** *A survey of fake news: Fundamental theories, detection methods, and opportunities.* ACM Computing Surveys, 53(5), 1–40.
  → Survey más reciente. Buen mapa del campo para la introducción del reporte.

- **Wang, W. Y. (2017).** *"Liar, liar pants on fire": A new benchmark dataset for fake news detection.* ACL 2017.
  → Paper que introduce el **LIAR dataset** que ya está en la propuesta. Reporta baselines sobre los que hay que comparar (best F1 ≈ 0.27, hay mucho margen).

- **Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018).** *Automatic detection of fake news.* COLING 2018.
  → Demuestra que features léxicos simples (n-gramas, LIWC) llegan a ~76% accuracy. Útil para defender por qué TF-IDF + LogReg es un baseline fuerte.

---

## 3. Embeddings semánticos

- **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* EMNLP 2019.
  → **Paper del modelo que usa VERITAS** (`sentence-transformers`). Cita obligatoria.

- **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).** *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL 2019.
  → Base de BERT y derivados como MiniLM.

- **Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020).** *MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers.* NeurIPS 2020.
  → Paper del modelo MiniLM-L6 (el que usa VERITAS como backbone). Justifica el balance velocidad/calidad.

---

## 4. Recuperación neuronal y reranking

- **Karpukhin, V., Oğuz, B., Min, S., et al. (2020).** *Dense Passage Retrieval for open-domain question answering.* EMNLP 2020.
  → **DPR.** Propone el enfoque dual-encoder que VERITAS adopta para la búsqueda semántica.

- **Khattab, O., & Zaharia, M. (2020).** *ColBERT: Efficient and effective passage search via contextualized late interaction over BERT.* SIGIR 2020.
  → Late-interaction. Más caro pero supera a DPR. Mencionable como trabajo futuro.

- **Nogueira, R., & Cho, K. (2019).** *Passage re-ranking with BERT.* arXiv:1901.04085.
  → **Justifica el reranker cross-encoder.** Two-stage retrieval (BM25 → BERT reranker) es estado del arte y casi free win sobre BM25 solo.

- **Lin, J., Nogueira, R., & Yates, A. (2021).** *Pretrained transformers for text ranking: BERT and beyond.* Synthesis Lectures on Human Language Technologies, 14(4), 1–325.
  → Libro/survey. Capítulo 3 cubre exactamente el pipeline two-stage de VERITAS.

---

## 5. Verificación de afirmaciones y evidencia

- **Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018).** *FEVER: a large-scale dataset for fact extraction and verification.* NAACL 2018.
  → **Paper clave para el nombre VERITAS.** Define las 3 etiquetas SUPPORTS/REFUTES/NOT ENOUGH INFO y el pipeline document retrieval → sentence selection → claim verification.

- **Hanselowski, A., Stab, C., Schulz, C., Li, Z., & Gurevych, I. (2019).** *A richly annotated corpus for different tasks in automated fact-checking.* CoNLL 2019.
  → Buena referencia complementaria a FEVER para fact-checking real-world.

- **Augenstein, I., Lioma, C., Wang, D., et al. (2019).** *MultiFC: A real-world multi-domain dataset for evidence-based fact checking of claims.* EMNLP 2019.
  → Dataset con 36k claims y evidencia. Más realista que FEVER (que se basa en Wikipedia).

- **Guo, Z., Schlichtkrull, M., & Vlachos, A. (2022).** *A survey on automated fact-checking.* TACL 10, 178–206.
  → **Survey reciente y muy citado.** Cubre claim detection, evidence retrieval, verdict prediction y justification production — exactamente el pipeline que VERITAS implementa parcialmente.

---

## 6. NLI / Entailment para verificación

- **Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015).** *A large annotated corpus for learning natural language inference.* EMNLP 2015 (SNLI).
- **Williams, A., Nangia, N., & Bowman, S. R. (2018).** *A broad-coverage challenge corpus for sentence understanding through inference.* NAACL 2018 (MNLI).
  → Datasets sobre los que está entrenado el modelo de NLI que VERITAS reusa
    (e.g. `cross-encoder/nli-deberta-v3-base`).

---

## 7. Evaluación de RI

- **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval.* Cambridge University Press.
  → Libro canónico del curso. Capítulo 8 (Evaluation) cubre P@K, R, MAP, NDCG.

- **Järvelin, K., & Kekäläinen, J. (2002).** *Cumulated gain-based evaluation of IR techniques.* ACM TOIS, 20(4), 422–446.
  → Paper original de **NDCG**. Cita obligatoria al usar la métrica.

---

## 8. RAG y futuro de la verificación

- **Lewis, P., Perez, E., Piktus, A., et al. (2020).** *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 2020.
  → Si en el futuro VERITAS agregara un LLM que explique el veredicto, este es el paradigma de referencia.

- **Atanasova, P., Simonsen, J. G., Lioma, C., & Augenstein, I. (2020).** *Generating fact checking explanations.* ACL 2020.
  → Cómo generar **justificaciones legibles** del veredicto. Mejora la "claridad" pedida por la rúbrica.

---

## Sugerencia para el reporte

Estructura recomendada de la sección de "Trabajo relacionado":

1. **Retrieval clásico** → cita §1 (Salton, Robertson).
2. **Detección de fake news** → cita §2 (Shu et al., Wang).
3. **Retrieval neuronal** → cita §3 y §4 (Reimers, Karpukhin, Nogueira).
4. **Verificación con evidencia** → cita §5 (FEVER, Guo et al. survey).
5. **Qué hace diferente VERITAS** → integración de los tres en un único prototipo
   académico con interfaz interactiva.
