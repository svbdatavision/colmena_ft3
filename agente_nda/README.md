# Agente IA para Clasificación de NDAs

Prototipo funcional (MVP) en **Python + Streamlit** que clasifica un documento NDA
por similitud semántica léxica frente a plantillas de referencia, **sin usar OpenAI**.

## Objetivo

Demostrar el concepto: un usuario sube un NDA (PDF o DOCX) y la aplicación
recomienda el modelo más parecido de la carpeta `docs/`, junto con un ranking
y un gráfico de similitudes.

## Cómo funciona

1. Se lee el documento cargado (PDF con `pdfplumber`, DOCX con `python-docx`).
2. Se leen automáticamente todos los modelos de la carpeta `docs/`.
3. Se vectorizan los textos con **TF-IDF** (`scikit-learn`).
4. Se calcula **cosine similarity** entre el NDA y cada modelo.
5. Se elige el modelo más similar y se muestran métricas, ranking (top 6) y gráfico.

## Estructura del proyecto

```text
agente_nda/
├── app.py              # Interfaz Streamlit
├── utils.py            # Extracción de texto y motor de clasificación
├── requirements.txt    # Dependencias
├── README.md           # Este archivo
└── docs/               # Modelos / plantillas NDA de referencia
    ├── MODELO SIMPLIFICADO ACUERDO DE CONFIDENCIALIDAD unilateral.docx
    ├── ACUERDO DE CONFIDENCIALIDAD SIMPLIFICADO USO EXCLUSIVO PROCUREMENT.docx
    ├── ACUERDO DE CONFIDENCIALIDAD SIMPLIFICADO USO EXCLUSIVO PROCUREMENT webdox.docx
    ├── D1-ModeloROBUSTONDABilateral-Octubre2023.docx
    └── ...
```

## Requisitos

- Python 3.9+
- pip

## Instalación y ejecución

Desde la carpeta `agente_nda/`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

La interfaz se abrirá en el navegador (por defecto `http://localhost:8501`).

## Uso

1. Coloque (o reemplace) las plantillas NDA en `docs/` (PDF o DOCX).
2. Ejecute la aplicación.
3. Suba un NDA de prueba.
4. Pulse **Analizar Documento**.
5. Revise:
   - Modelo recomendado
   - Nivel de confianza / porcentaje de similitud
   - Tipo detectado
   - Justificación
   - Ranking de los 6 modelos
   - Gráfico de barras horizontal con todas las similitudes

## Notas del prototipo

- Es un MVP de demostración: la clasificación se basa en similitud de vocabulario,
  no en un LLM ni en reglas jurídicas exhaustivas.
- Documentos escaneados solo como imagen (sin capa de texto) no podrán analizarse
  sin un paso previo de OCR.
- Cuantos más modelos reales haya en `docs/`, más útil será la recomendación.
