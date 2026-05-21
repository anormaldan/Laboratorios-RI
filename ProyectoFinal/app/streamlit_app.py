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

# ---------------------------------------------------------------------------
# Consultas sugeridas por tema — basadas en el análisis del corpus real
# (Fake and Real News Dataset 2015-2017 + LIAR dataset)
#
# Cada entrada: (etiqueta_corta, query_completa, resultado_esperado)
#   resultado_esperado: "FAKE" | "REAL" | "?"  (solo orientativo)
# ---------------------------------------------------------------------------
SUGGESTED_QUERIES: dict[str, list[tuple[str, str, str]]] = {
    "🏛️ Política US": [
        ("Fraude electoral",        "election fraud claims without evidence",                 "FAKE"),
        ("Trump y sus negocios",    "Trump campaign paid millions to his own businesses",      "FAKE"),
        ("Recorte de impuestos",    "Republican tax cuts benefit the wealthy Americans",       "REAL"),
        ("Obamacare derogado",      "Republicans vote to repeal Obamacare health care",        "REAL"),
        ("Obama y voto ilegal",     "Obama told illegal aliens it was okay to vote",           "FAKE"),
        ("Mueller investigación",   "Mueller improperly obtained transition documents Russia", "FAKE"),
    ],
    "🌍 Geopolítica": [
        ("Rusia / elecciones",      "Russia interference in United States election probe",     "REAL"),
        ("Corea del Norte",         "North Korea nuclear weapons missile test",                "REAL"),
        ("Acuerdo con Irán",        "Iran nuclear deal negotiations United States sanctions",  "REAL"),
        ("Brexit",                  "Brexit vote United Kingdom European Union",               "REAL"),
        ("Siria guerra civil",      "Syria civil war chemical weapons Assad",                  "REAL"),
        ("China comercio",          "China trade deal tariffs United States",                  "REAL"),
    ],
    "🏥 Salud y ciencia": [
        ("Vacunas y autismo",       "vaccines cause autism in children fake report",           "FAKE"),
        ("Reforma de salud",        "health care reform legislation Obamacare mandate",        "?"),
        ("Cambio climático",        "climate change is supported by scientific consensus",     "REAL"),
        ("Cirugías gratuitas",      "Obamacare mandate free sex change surgeries",             "FAKE"),
    ],
    "🔫 Crimen e inmigración": [
        ("Crimen e indocumentados", "illegal aliens crime rates steal jobs Americans",         "FAKE"),
        ("Control de armas",        "gun control legislation Senate vote Republicans",         "REAL"),
        ("Veda de visados",         "Muslim ban executive order immigration court",            "?"),
        ("Berkeley disturbios",     "Berkeley rioters chase beat people",                     "FAKE"),
    ],
    "⚡ Virales / conspiración": [
        ("Sharia en EEUU",          "Democrats implement sharia law in America",               "FAKE"),
        ("CDC autodestrucción",     "CDC offices will self-destruct catastrophic event",       "FAKE"),
        ("Pokemon vigilancia",      "Pokemon Go surveillance capitalism society",              "FAKE"),
        ("Bernie vs Trump",         "Bernie Sanders only candidate who beats Trump poll",      "FAKE"),
    ],
}


def render_result(r) -> None:
    color = "#2e7d32" if r.verdict == "REAL" else "#c62828"
    bg    = "#e8f5e9" if r.verdict == "REAL" else "#ffebee"

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
                score: <b>{r.score:.3f}</b> &nbsp;·&nbsp; P(real): <b>{r.prob_real:.2f}</b>
            </div>
            <div style='margin-top:8px;'>{r.snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggested_queries() -> None:
    """Muestra chips de consultas sugeridas agrupadas por tema."""
    with st.expander("💡 Consultas sugeridas — basadas en el corpus (2015-2017)", expanded=True):
        st.caption(
            "El corpus contiene noticias políticas y de actualidad internacional de 2015-2017 "
            "(Fake and Real News Dataset + LIAR). Haz clic en cualquier consulta para cargarla."
        )

        for category, queries in SUGGESTED_QUERIES.items():
            st.markdown(f"**{category}**")
            # Chips en filas de 3
            cols = st.columns(3)
            for idx, (label, query_text, expected) in enumerate(queries):
                badge = {"FAKE": "🔴", "REAL": "🟢", "?": "❓"}.get(expected, "")
                btn_label = f"{badge} {label}"
                if cols[idx % 3].button(btn_label, key=f"sugg_{category}_{idx}", use_container_width=True):
                    st.session_state["_query_text"] = query_text
                    st.rerun()
            st.markdown("")   # espacio entre categorías


def main() -> None:
    # ── session state para la query ─────────────────────────────────────────
    if "_query_text" not in st.session_state:
        st.session_state["_query_text"] = ""

    # ── cabecera ─────────────────────────────────────────────────────────────
    st.title("⚖️ VERITAS")
    st.caption(
        "**Verification and Retrieval Intelligent Truth Analysis System** — "
        "Proyecto final del curso de Recuperación de Información. "
        "Combina TF-IDF / BM25 / embeddings con verificación de noticias. "
        "_Equipo: Jonathan Tiro · Alejandro Nava · Andrea Ortega._"
    )

    # ── sidebar ───────────────────────────────────────────────────────────────
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
            "**Convenciones del corpus**\n\n"
            "Noticias en inglés, 2015-2017\n\n"
            "**Temas principales:**\n"
            "- Política US (elecciones, Congreso)\n"
            "- Geopolítica (Rusia, NK, Irán, Brexit)\n"
            "- Salud y ciencia\n"
            "- Inmigración y crimen\n\n"
            "**Clasificación:**\n\n"
            "🟢 REAL · 🔴 FAKE\n\n"
            "**FEVER (si activado):**\n"
            "- 🟩 SUPPORTS — evidencia respalda\n"
            "- 🟥 REFUTES — evidencia contradice\n"
            "- ⬜ NEI — info insuficiente"
        )
        st.markdown("---")
        st.markdown(
            "**Datasets:**\n"
            "- Fake & Real News (Kaggle, ~45k artículos)\n"
            "- LIAR (PolitiFact, ~12k afirmaciones)\n\n"
            "**Rango temporal:** 2015–2017"
        )
        st.markdown("---")
        st.caption("**Equipo:** Jonathan Tiro · Alejandro Nava · Andrea Ortega")

    # ── pipeline ─────────────────────────────────────────────────────────────
    pipeline = get_pipeline(model, sample, use_reranker, use_verifier)

    # ── consultas sugeridas ───────────────────────────────────────────────────
    render_suggested_queries()

    # ── input de búsqueda ─────────────────────────────────────────────────────
    query = st.text_input(
        "Escribe una afirmación o titular para verificar",
        value=st.session_state["_query_text"],
        placeholder="Ej. vaccines cause autism in children",
        key="query_input",
    )
    # Sincroniza: si el usuario escribe manualmente, actualizamos session_state
    st.session_state["_query_text"] = query

    # ── resultados ────────────────────────────────────────────────────────────
    if query:
        with st.spinner("Buscando y verificando…"):
            results = pipeline.search_and_verify(query, top_k=top_k)

        if not results:
            st.warning("Sin resultados para esa consulta.")
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
