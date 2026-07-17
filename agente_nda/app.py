"""
Agente IA para Clasificación de NDAs — Prototipo MVP (Streamlit).

Cómo ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

La aplicación compara un NDA cargado (PDF/DOCX) contra los modelos
de la carpeta docs/ usando TF-IDF y cosine similarity (sin OpenAI).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    DEFAULT_DOCS_DIR,
    build_similarity_dataframe,
    classify_document,
    extract_text_from_file,
    load_model_documents,
)


# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agente IA para Clasificación de NDAs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_custom_styles() -> None:
    """Aplica estilos CSS ligeros para un look limpio y profesional."""
    st.markdown(
        """
        <style>
            /* Tipografía y fondo general */
            .stApp {
                background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
            }
            .main-header {
                font-size: 2.1rem;
                font-weight: 700;
                color: #1a365d;
                margin-bottom: 0.25rem;
            }
            .main-subtitle {
                color: #4a5568;
                font-size: 1.05rem;
                margin-bottom: 1.5rem;
            }
            /* Tarjeta de resultado */
            .result-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 4px 14px rgba(26, 54, 93, 0.06);
                margin-top: 0.5rem;
                margin-bottom: 1rem;
            }
            .result-card h3 {
                color: #1a365d;
                margin-top: 0;
                margin-bottom: 0.75rem;
            }
            .result-label {
                color: #718096;
                font-size: 0.85rem;
                margin-bottom: 0.15rem;
            }
            .result-value {
                color: #2d3748;
                font-size: 1.05rem;
                font-weight: 600;
                margin-bottom: 0.85rem;
            }
            .justification {
                color: #4a5568;
                font-size: 0.95rem;
                line-height: 1.45;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_reference_models(docs_dir: str) -> dict:
    """
    Carga y cachea los textos de los modelos de la carpeta docs/.

    Se usa cache_resource para no re-leer los archivos en cada interacción.
    """
    return load_model_documents(docs_dir)


def render_header() -> None:
    """Dibuja el título y la descripción solicitados."""
    st.markdown(
        '<p class="main-header">Agente IA para Clasificación de NDAs</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="main-subtitle">'
        "Suba un documento NDA para identificar automáticamente su tipología "
        "y recomendar el modelo correspondiente."
        "</p>",
        unsafe_allow_html=True,
    )


def render_sidebar_models(model_names: list[str]) -> None:
    """Muestra en el sidebar los modelos detectados en docs/."""
    with st.sidebar:
        st.header("Modelos de referencia")
        st.caption(f"Carpeta: `{DEFAULT_DOCS_DIR.name}/`")
        st.write(f"**{len(model_names)}** documentos cargados:")
        for name in model_names:
            st.markdown(f"- {name}")


def render_result_card(result: dict) -> None:
    """Tarjeta con modelo recomendado, confianza, tipo y justificación."""
    st.markdown(
        f"""
        <div class="result-card">
            <h3>Resultado del análisis</h3>
            <div class="result-label">Modelo recomendado</div>
            <div class="result-value">{result["modelo_recomendado"]}</div>
            <div class="result-label">Nivel de confianza</div>
            <div class="result-value">{result["nivel_confianza"]}
                ({result["similitud_pct"]:.2f}%)</div>
            <div class="result-label">Tipo detectado</div>
            <div class="result-value">{result["tipo_detectado"]}</div>
            <div class="result-label">Justificación</div>
            <div class="justification">{result["justificacion"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(result: dict) -> None:
    """Métricas rápidas con st.metric y barra de progreso."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Similitud máxima",
            value=f"{result['similitud_pct']:.2f}%",
        )
    with col2:
        st.metric(
            label="Nivel de confianza",
            value=result["nivel_confianza"],
        )
    with col3:
        st.metric(
            label="Tipo detectado",
            value=result["tipo_detectado"],
        )

    # Barra de progreso normalizada 0–1.
    st.caption("Confianza del modelo recomendado")
    st.progress(min(max(result["similitud"] , 0.0), 1.0))


def render_ranking_table(result: dict) -> None:
    """Tabla con el ranking de los 6 modelos más similares."""
    st.subheader("Ranking de los 6 modelos")
    df = build_similarity_dataframe(result["ranking"])
    st.dataframe(
        df[["Modelo completo", "Tipo", "Similitud (%)"]],
        use_container_width=True,
        hide_index=True,
    )


def render_similarity_chart(result: dict) -> None:
    """
    Gráfico de barras horizontal con la similitud de todos los modelos.
    """
    st.subheader("Similitud con todos los modelos")

    full_ranking = result["ranking_completo"]
    # Invertimos para que el más similar quede arriba en el gráfico.
    labels = [item["modelo"] for item in reversed(full_ranking)]
    values = [item["similitud_pct"] for item in reversed(full_ranking)]

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(labels) + 1)))
    bars = ax.barh(labels, values, color="#2b6cb0", edgecolor="none")

    # Resalta la barra del modelo recomendado.
    if bars:
        bars[-1].set_color("#38a169")

    ax.set_xlabel("Similitud (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Comparación TF-IDF / Cosine Similarity")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 1.5, 98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            color="#2d3748",
        )

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    inject_custom_styles()
    render_header()

    # --- Carga de modelos de referencia ---
    try:
        model_texts = get_reference_models(str(DEFAULT_DOCS_DIR))
    except Exception as exc:  # noqa: BLE001
        st.error(
            "No fue posible cargar los modelos de la carpeta `docs/`.\n\n"
            f"**Detalle:** {exc}"
        )
        st.stop()

    render_sidebar_models(list(model_texts.keys()))

    st.info(
        f"Se cargaron **{len(model_texts)}** modelos desde la carpeta `docs/`. "
        "Suba un NDA en PDF o DOCX y pulse **Analizar Documento**."
    )

    # --- Uploader ---
    uploaded_file = st.file_uploader(
        "Cargar documento NDA",
        type=["pdf", "docx"],
        help="Formatos admitidos: PDF y DOCX.",
    )

    analyze_clicked = st.button(
        "Analizar Documento",
        type="primary",
        use_container_width=False,
    )

    if not analyze_clicked:
        st.caption(
            "El análisis utiliza vectorización TF-IDF y similitud coseno. "
            "No se envían datos a servicios externos (sin OpenAI)."
        )
        return

    # --- Validaciones previas al análisis ---
    if uploaded_file is None:
        st.warning("Por favor, cargue un archivo PDF o DOCX antes de analizar.")
        return

    # --- Análisis con spinner ---
    with st.spinner("Analizando documento y comparando con los modelos..."):
        try:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_text = extract_text_from_file(
                uploaded_bytes,
                filename=uploaded_file.name,
            )
            result = classify_document(
                uploaded_text=uploaded_text,
                model_texts=model_texts,
                top_n=6,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error durante el análisis: {exc}")
            return

    # --- Resultados ---
    st.success(
        f"Análisis completado. Modelo recomendado: "
        f"**{result['modelo_recomendado']}** "
        f"({result['similitud_pct']:.2f}% de similitud)."
    )

    render_metrics(result)
    render_result_card(result)
    render_ranking_table(result)
    render_similarity_chart(result)

    # Detalle opcional del texto extraído (útil en demos).
    with st.expander("Ver extracto del texto analizado"):
        preview = uploaded_text[:2500]
        if len(uploaded_text) > 2500:
            preview += "\n\n[... texto truncado ...]"
        st.text(preview)


if __name__ == "__main__":
    main()
