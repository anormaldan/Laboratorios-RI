"""Demo Streamlit del sistema VERITAS — RI + detección de fake news.

Ejecutar:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import EvidenceConsensus, SearchPipeline, SearchResult


st.set_page_config(
    page_title="VERITAS — Truth Analysis System",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache del pipeline (se reconstruye solo si cambia algún parámetro)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando corpus e índices…")
def get_pipeline(
    model: str,
    sample: int,
    use_reranker: bool,
    use_verifier: bool,
    use_transformer_clf: bool,
) -> SearchPipeline:
    return SearchPipeline.build(
        model=model,
        sample_size=sample,
        use_reranker=use_reranker,
        use_verifier=use_verifier,
        use_transformer_clf=use_transformer_clf,
    )


# ---------------------------------------------------------------------------
# Paletas de color
# ---------------------------------------------------------------------------

FEVER_COLORS = {
    "SUPPORTS": ("#1b5e20", "#c8e6c9"),
    "REFUTES":  ("#b71c1c", "#ffcdd2"),
    "NEI":      ("#37474f", "#cfd8dc"),
}

VERDICT_COLORS = {
    "REAL": ("#2e7d32", "#e8f5e9"),
    "FAKE": ("#c62828", "#ffebee"),
}

# ---------------------------------------------------------------------------
# Consultas sugeridas — basadas en el corpus real (2015-2017)
# ---------------------------------------------------------------------------

SUGGESTED_QUERIES: dict[str, list[tuple[str, str, str]]] = {
    "🏛️ Política US": [
        ("Fraude electoral",     "election fraud claims without evidence",                "FAKE"),
        ("Trump y negocios",     "Trump campaign paid millions to his own businesses",     "FAKE"),
        ("Recorte impuestos",    "Republican tax cuts benefit the wealthy Americans",      "REAL"),
        ("Obamacare derogado",   "Republicans vote to repeal Obamacare health care",       "REAL"),
        ("Obama voto ilegal",    "Obama told illegal aliens it was okay to vote",          "FAKE"),
        ("Mueller / Rusia",      "Mueller improperly obtained transition documents Russia","FAKE"),
    ],
    "🌍 Geopolítica": [
        ("Rusia / elecciones",   "Russia interference in United States election probe",    "REAL"),
        ("Corea del Norte",      "North Korea nuclear weapons missile test",               "REAL"),
        ("Acuerdo con Irán",     "Iran nuclear deal negotiations United States",           "REAL"),
        ("Brexit",               "Brexit vote United Kingdom European Union",              "REAL"),
        ("Siria / química",      "Syria civil war chemical weapons Assad",                 "REAL"),
        ("China / comercio",     "China trade deal tariffs United States",                 "REAL"),
    ],
    "🏥 Salud y ciencia": [
        ("Vacunas y autismo",    "vaccines cause autism in children fake report",          "FAKE"),
        ("Reforma de salud",     "health care reform legislation Obamacare mandate",       "?"),
        ("Cambio climático",     "climate change is supported by scientific consensus",    "REAL"),
        ("Cirugías gratuitas",   "Obamacare mandate free sex change surgeries",            "FAKE"),
    ],
    "🔫 Crimen e inmigración": [
        ("Crimen / ilegales",    "illegal aliens crime rates steal jobs Americans",        "FAKE"),
        ("Control de armas",     "gun control legislation Senate vote Republicans",        "REAL"),
        ("Veda de visados",      "Muslim ban executive order immigration court",           "?"),
        ("Berkeley disturbios",  "Berkeley rioters chase beat people",                     "FAKE"),
    ],
    "⚡ Virales / conspiración": [
        ("Sharia en EEUU",       "Democrats implement sharia law in America",              "FAKE"),
        ("CDC autodestrucción",  "CDC offices will self-destruct catastrophic event",      "FAKE"),
        ("Pokemon vigilancia",   "Pokemon Go surveillance capitalism society",             "FAKE"),
        ("Bernie vs Trump",      "Bernie Sanders only candidate who beats Trump poll",     "FAKE"),
    ],
}


# ---------------------------------------------------------------------------
# Renderizado de un resultado individual
# ---------------------------------------------------------------------------

def render_result(r: SearchResult) -> None:
    color, bg = VERDICT_COLORS[r.verdict]

    # Badge FEVER
    fever_badge = ""
    if r.fever_verdict:
        fc, fbg = FEVER_COLORS[r.fever_verdict]
        fever_badge = (
            f"<span style='background:{fbg}; color:{fc}; padding:2px 8px;"
            f"border-radius:10px; font-size:0.78em; margin-left:8px; font-weight:600;'>"
            f"FEVER: {r.fever_verdict}</span>"
        )

    # Badge ensemble (si activo)
    ensemble_badge = ""
    if r.combined_prob_real is not None:
        delta = r.combined_prob_real - r.prob_real
        sign = "▲" if delta >= 0 else "▼"
        ensemble_badge = (
            f"<span style='background:#e3f2fd; color:#0d47a1; padding:2px 8px;"
            f"border-radius:10px; font-size:0.78em; margin-left:6px;'>"
            f"ensemble {sign}{abs(delta):.2f}</span>"
        )

    # Barra de confianza
    pct = int(r.confidence * 200)   # [0, 100]
    bar_color = color

    st.markdown(
        f"""
        <div style='border-left:6px solid {color}; background:{bg};
                    padding:12px 16px; border-radius:6px; margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <strong style='font-size:1.02em;'>[{r.rank}] {r.title or "(sin título)"}</strong>
                <span style='white-space:nowrap;'>
                    <span style='color:{color}; font-weight:bold;'>{r.verdict}</span>
                    {fever_badge}{ensemble_badge}
                </span>
            </div>
            <div style='font-size:0.85em; color:#555; margin-top:5px;'>
                score: <b>{r.score:.3f}</b> &nbsp;·&nbsp;
                P(real): <b>{r.prob_real:.2f}</b>
                {"&nbsp;·&nbsp; P(real) combinado: <b>" + f"{r.combined_prob_real:.2f}</b>"
                  if r.combined_prob_real is not None else ""}
                &nbsp;·&nbsp; confianza:
                <span style='display:inline-block; background:#ddd; border-radius:4px;
                             width:80px; height:8px; vertical-align:middle; margin-left:4px;'>
                    <span style='display:block; background:{bar_color}; border-radius:4px;
                                 width:{pct}%; height:8px;'></span>
                </span>
            </div>
            <div style='margin-top:8px; font-size:0.93em;'>{r.snippet}</div>
            {"<div style='margin-top:6px; font-size:0.8em; color:#666; border-top:1px solid #ddd; padding-top:4px;'>"
             "<b>Evidencia usada en NLI:</b> " + (r.evidence_sentence or "") + "</div>"
             if r.evidence_sentence else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Panel de consenso de evidencia
# ---------------------------------------------------------------------------

def render_consensus(consensus: EvidenceConsensus) -> None:
    """Muestra el resumen agregado de veredictos FEVER."""
    total = consensus.total or 1
    sup_pct = int(consensus.support_ratio * 100)
    ref_pct = int(consensus.refute_ratio * 100)
    nei_pct = 100 - sup_pct - ref_pct

    dom_color = {"SUPPORTS": "#2e7d32", "REFUTES": "#b71c1c", "NEI": "#37474f"}
    dom = consensus.dominant
    color = dom_color.get(dom, "#37474f")

    st.markdown("#### 🧭 Consenso de evidencia recuperada")
    st.markdown(
        f"De los **{consensus.total}** documentos recuperados, "
        f"**{consensus.supports}** SOPORTAN el claim, "
        f"**{consensus.refutes}** lo REFUTAN y "
        f"**{consensus.nei}** tienen información insuficiente. "
        f"Veredicto dominante: <span style='color:{color}; font-weight:bold;'>"
        f"{dom}</span>",
        unsafe_allow_html=True,
    )

    # Barra segmentada
    st.markdown(
        f"""
        <div style='display:flex; height:16px; border-radius:8px; overflow:hidden;
                    margin:8px 0 4px 0; width:100%;'>
            <div style='background:#2e7d32; width:{sup_pct}%; transition:.3s;'></div>
            <div style='background:#b71c1c; width:{ref_pct}%; transition:.3s;'></div>
            <div style='background:#90a4ae; width:{nei_pct}%; transition:.3s;'></div>
        </div>
        <div style='font-size:0.78em; color:#555;'>
            <span style='color:#2e7d32;'>■</span> SUPPORTS {sup_pct}% &nbsp;
            <span style='color:#b71c1c;'>■</span> REFUTES {ref_pct}% &nbsp;
            <span style='color:#90a4ae;'>■</span> NEI {nei_pct}%
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Panel de consultas sugeridas
# ---------------------------------------------------------------------------

def render_suggested_queries() -> None:
    with st.expander(
        "💡 Consultas sugeridas — basadas en el corpus (2015-2017)", expanded=True
    ):
        st.caption(
            "El corpus contiene noticias políticas e internacionales de 2015-2017 "
            "(Fake and Real News Dataset + LIAR). Clic en cualquier chip para cargarla."
        )
        for category, queries in SUGGESTED_QUERIES.items():
            st.markdown(f"**{category}**")
            cols = st.columns(3)
            for idx, (label, query_text, expected) in enumerate(queries):
                badge = {"FAKE": "🔴", "REAL": "🟢", "?": "❓"}.get(expected, "")
                if cols[idx % 3].button(
                    f"{badge} {label}",
                    key=f"sugg_{category}_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["_query_text"] = query_text
                    st.rerun()
            st.markdown("")


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main() -> None:
    # Inicializar session state
    if "_query_text" not in st.session_state:
        st.session_state["_query_text"] = ""

    # ── Cabecera ─────────────────────────────────────────────────────────
    st.title("⚖️ VERITAS")
    st.caption(
        "**Verification and Retrieval Intelligent Truth Analysis System** — "
        "Proyecto final del curso de Recuperación de Información. "
        "Combina TF-IDF / BM25 / embeddings con verificación de noticias. "
        "_Equipo: Jonathan Tiro · Alejandro Nava · Andrea Ortega._"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")
        model = st.selectbox(
            "Modelo de recuperación",
            ["bm25", "tfidf", "semantic"],
            index=0,
            help="BM25: probabilístico clásico (Robertson & Zaragoza, 2009)\n"
                 "TF-IDF: vectorial léxico (Salton, 1975)\n"
                 "Semantic: dense retrieval con MiniLM (Reimers, 2019)",
        )
        top_k = st.slider("Top-K resultados", 1, 20, 5)
        sample = st.select_slider(
            "Tamaño del corpus",
            options=[1000, 2000, 5000, 10000, 20000],
            value=5000,
        )

        st.markdown("---")
        st.subheader("🔬 Mejoras avanzadas")

        use_reranker = st.checkbox(
            "Cross-Encoder Reranker",
            help="Nogueira & Cho (2019). Reordena top-100 con ms-marco-MiniLM.",
        )
        use_verifier = st.checkbox(
            "Verificación FEVER (NLI)",
            help="Thorne et al. (2018). SUPPORTS / REFUTES / NEI por documento.",
        )
        use_transformer_clf = st.checkbox(
            "Clasificador Transformer (ALBERT)",
            help=(
                "Azizah et al. (2023) — ALBERT alcanza 87.6 % accuracy vs 56 % "
                "de NB+TF-IDF (Sutradhar et al., 2023). Fine-tuning en el corpus. "
                "Requiere: pip install torch transformers. "
                "Primera vez: entrena ~5-20 min (según hardware)."
            ),
        )

        st.markdown("---")
        st.markdown(
            "**Convenciones del corpus**\n\n"
            "Noticias en inglés · 2015-2017\n\n"
            "**Temas:** política US, geopolítica, salud, inmigración\n\n"
            "**Clasificación binaria:**\n"
            "🟢 REAL · 🔴 FAKE\n\n"
            "**FEVER (si activado):**\n"
            "- 🟩 SUPPORTS — evidencia respalda\n"
            "- 🟥 REFUTES — evidencia contradice\n"
            "- ⬜ NEI — info insuficiente\n\n"
            "**Ensemble:**\n"
            "Combina P(real) del clasificador con señal NLI (peso 35 %)."
        )
        st.markdown("---")
        st.markdown(
            "**Datasets:**\n"
            "- Fake & Real News (Kaggle, ~45k)\n"
            "- LIAR (PolitiFact, ~12k)\n\n"
            "**Rango temporal:** 2015–2017"
        )
        st.markdown("---")
        st.caption("Jonathan Tiro · Alejandro Nava · Andrea Ortega")

    # ── Pipeline ─────────────────────────────────────────────────────────
    pipeline = get_pipeline(
        model, sample, use_reranker, use_verifier, use_transformer_clf
    )

    # ── Consultas sugeridas ───────────────────────────────────────────────
    render_suggested_queries()

    # ── Input de búsqueda ─────────────────────────────────────────────────
    query = st.text_input(
        "Escribe una afirmación o titular para verificar",
        value=st.session_state["_query_text"],
        placeholder="Ej. vaccines cause autism in children",
        key="query_input",
    )
    st.session_state["_query_text"] = query

    # ── Resultados ────────────────────────────────────────────────────────
    if query:
        with st.spinner("Buscando y verificando…"):
            results, consensus = pipeline.search_and_verify(query, top_k=top_k)

        if not results:
            st.warning("Sin resultados para esa consulta.")
            return

        # Métricas rápidas
        n_fake = sum(1 for r in results if r.verdict == "FAKE")
        n_real = len(results) - n_fake
        avg_conf = sum(r.confidence for r in results) / len(results)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Resultados", len(results))
        col2.metric("🔴 FAKE", n_fake)
        col3.metric("🟢 REAL", n_real)
        col4.metric("Confianza media", f"{avg_conf:.2f}")
        if use_verifier and consensus:
            col5.metric(
                "Consenso FEVER",
                f"{consensus.supports}S / {consensus.refutes}R",
                help="S=SUPPORTS, R=REFUTES",
            )
        else:
            col5.metric("FEVER", "off")

        # Consenso visual
        if use_verifier and consensus:
            render_consensus(consensus)

        # Resultados individuales
        st.markdown("### Resultados")
        for r in results:
            render_result(r)


if __name__ == "__main__":
    main()
