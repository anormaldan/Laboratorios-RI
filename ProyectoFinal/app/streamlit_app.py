"""Demo Streamlit del sistema de RI + detección de fake news.

Ejecutar:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import SearchPipeline


st.set_page_config(
    page_title="VERITAS — Truth Analysis System",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando corpus e índices…")
def get_pipeline(model: str, sample: int, use_reranker: bool, use_verifier: bool) -> SearchPipeline:
    return SearchPipeline.build(
        model=model,
        sample_size=sample,
        use_reranker=use_reranker,
        use_verifier=use_verifier,
    )


FEVER_COLORS = {
    "SUPPORTS": ("#1b5e20", "#c8e6c9"),
    "REFUTES":  ("#b71c1c", "#ffcdd2"),
    "NEI":      ("#37474f", "#cfd8dc"),
}


def render_result(r):
    color = "#2e7d32" if r.verdict == "REAL" else "#c62828"
    bg = "#e8f5e9" if r.verdict == "REAL" else "#ffebee"

    fever_badge = ""
    if r.fever_verdict:
        fc, fbg = FEVER_COLORS[r.fever_verdict]
        fever_badge = (
            f"<span style='background:{fbg}; color:{fc}; padding:2px 8px;"
            f"border-radius:10px; font-size:0.8em; margin-left:8px;'>"
            f"FEVER: {r.fever_verdict}</span>"
        )

    st.markdown(
        f"""
        <div style='border-left:6px solid {color}; background:{bg};
                    padding:12px 16px; border-radius:6px; margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between;'>
                <strong style='font-size:1.05em;'>[{r.rank}] {r.title}</strong>
                <span style='color:{color}; font-weight:bold;'>{r.verdict}{fever_badge}</span>
            </div>
            <div style='font-size:0.9em; color:#555; margin-top:4px;'>
                score: <b>{r.score:.3f}</b> · P(real): <b>{r.prob_real:.2f}</b>
            </div>
            <div style='margin-top:8px;'>{r.snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("⚖️ VERITAS")
    st.caption(
        "**Verification and Retrieval Intelligent Truth Analysis System** — "
        "Proyecto final del curso de Recuperación de Información. "
        "Combina TF-IDF / BM25 / embeddings con verificación de noticias. "
        "_Equipo: Jonathan Tiro · Alejandro Nava · Andrea Ortega._"
    )

    with st.sidebar:
        st.header("Configuración")
        model = st.selectbox("Modelo de recuperación", ["bm25", "tfidf", "semantic"], index=0)
        top_k = st.slider("Top-K resultados", 1, 20, 5)
        sample = st.select_slider(
            "Tamaño del corpus",
            options=[1000, 2000, 5000, 10000, 20000],
            value=5000,
        )

        st.markdown("---")
        st.subheader("Mejoras avanzadas")
        use_reranker = st.checkbox(
            "Cross-Encoder Reranker",
            help="Nogueira & Cho (2019). Reordena top-100 con un BERT cross-encoder.",
        )
        use_verifier = st.checkbox(
            "Verificación FEVER (NLI)",
            help="Thorne et al. (2018). Etiqueta cada evidencia como SUPPORTS / REFUTES / NEI.",
        )

        st.markdown("---")
        st.markdown(
            "**Convenciones**\n\n"
            "🟢 REAL · 🔴 FAKE (clasificador binario)\n\n"
            "FEVER:\n"
            "- 🟩 SUPPORTS — evidencia respalda\n"
            "- 🟥 REFUTES — evidencia contradice\n"
            "- ⬜ NEI — info insuficiente"
        )
        st.markdown("---")
        st.caption(
            "**Equipo:** Jonathan Tiro · Alejandro Nava · Andrea Ortega"
        )

    pipeline = get_pipeline(model, sample, use_reranker, use_verifier)

    query = st.text_input(
        "Escribe una afirmación o titular para verificar",
        placeholder="Ej. vaccines cause autism in children",
    )

    if query:
        with st.spinner("Buscando y verificando…"):
            results = pipeline.search_and_verify(query, top_k=top_k)

        if not results:
            st.warning("Sin resultados.")
            return

        n_fake = sum(1 for r in results if r.verdict == "FAKE")
        n_real = len(results) - n_fake

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resultados", len(results))
        c2.metric("🔴 Posibles FAKE", n_fake)
        c3.metric("🟢 Posibles REAL", n_real)
        if use_verifier:
            n_refute = sum(1 for r in results if r.fever_verdict == "REFUTES")
            c4.metric("🟥 FEVER REFUTES", n_refute)
        else:
            c4.metric("FEVER", "off")

        st.markdown("### Resultados")
        for r in results:
            render_result(r)


if __name__ == "__main__":
    main()
