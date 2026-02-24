# INVENTARIO DE ARCHIVOS - FASTTRACK 3.0
**Fecha:** 2 de Octubre, 2025

---

## ARCHIVOS INCLUIDOS EN PRODUCCIÓN
### Archivo: `ft3_colmena_production_20251002_FINAL.tar.gz` (14 MB)

### 📂 Código Fuente (src/)
```
src/
├── __init__.py
├── data_loader.py          # Conexión y carga desde Snowflake
├── data_loader_v2.py       # Versión mejorada del loader
├── feature_engineering.py  # Transformación de 269 variables → ~500 features
├── model_training.py       # LightGBM + Optuna con almacenamiento persistente
├── model_auditor.py        # Auditoría y reportes de performance
└── information_value.py    # Cálculo de Information Value para features
```

### 🚀 Scripts de Ejecución
```
FT3_dia.py                  # Script principal de predicción diaria (pipeline automático)
FT30.py                     # Script de predicción con rango de fechas personalizado
main.py                     # Pipeline de entrenamiento y validación
run_daily_pipeline.py       # Orquestador del pipeline diario (Python)
run_daily_pipeline.sh       # Orquestador del pipeline diario (Bash)
Variables_cat_train.py      # Definición de las 269 variables por categoría (CRÍTICO)
```

### 📊 Scripts de Análisis y Optimización
```
analyze_threshold_optimization.py     # Análisis detallado de umbrales óptimos
optimize_with_compin.py               # Optimización de umbrales con costos COMPIN
show_optimal_thresholds_compin.py     # Visualización de umbrales recomendados
```

### 🐳 Docker
```
docker/
├── Dockerfile              # Imagen del contenedor FT3
└── crontab                 # Configuración de tareas programadas (6 AM diario)

docker-compose.yml          # Orquestación de servicios (scheduler, training, monitor)
```

### 🛠️ Scripts de Mantenimiento (scripts/)
```
scripts/
├── entrypoint.sh                     # Punto de entrada del contenedor
├── run_daily.sh                      # Script diario simplificado
├── run_weekly_validation.sh          # Validación semanal del modelo
├── run_monthly_update.sh             # Actualización mensual de variables
├── run_quarterly_training.sh         # Reentrenamiento trimestral
├── setup_cron_local.sh               # Configuración de cron local
├── setup_launchd.sh                  # Configuración de launchd (macOS)
└── com.fasttrack.ft3dia.plist        # LaunchDaemon para macOS
```

### 📊 Queries SQL
```
query_diaria.sql            # Carga diaria de licencias (Mar-Dom)
query_lunes.sql             # Carga especial lunes (incluye fin de semana)
query_2.sql                 # Actualización de MODELO_LM_202507_TRAIN
```

### 🤖 Modelos Pre-entrenados (models/)
```
models/
├── fasttrack_model.pkl                      # Modelo LightGBM entrenado
├── feature_fasttrack.pkl                    # Transformers (TF-IDF, Label Encoders)
├── optuna_study.db                          # Base de datos de optimización
└── fasttrack_model_feature_importance.csv   # Importancia de features
```

### ⚙️ Configuración
```
config.yaml                 # Configuración del modelo y Snowflake
requirements.txt            # Dependencias Python
.env.example                # Plantilla de variables de entorno
.gitignore                  # Archivos ignorados por git
```

### 📖 Documentación
```
README.md                           # Documentación técnica del proyecto
RESUMEN_EJECUTIVO_COLMENA.md        # Este documento - resumen ejecutivo completo
```

---

## ARCHIVOS MOVIDOS A old/ (No incluidos en producción)

### 🗂️ Versiones Antiguas del Modelo
```
old/
└── FT20.py                          # Versión 2.0 del modelo (obsoleta)
```

### 📊 Scripts de Análisis (No críticos)
```
old/
├── histogramas.py                           # Análisis exploratorio con visualizaciones
├── check_fields_simple.py                   # Verificación de campos
├── compare_queries.py                       # Comparación de queries SQL
└── analisis_histogramas/                    # Directorio de análisis históricos
```

### 📋 Definiciones de Tablas Snowflake (SQL)
```
old/
├── OPX.P_DDV_OPX_MDPREDICTIVO.AFILIADOS_METRICAS_MENSUALES.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.BASE_LM_PERTIAJES_PROPAGADOS.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.BOLETINES_ACCIONES.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.CPA_LM_BASE_AMPLIADA.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.FT30_PREDICCIONES_DIARIAS.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.LME_LIC.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN.sql
├── OPX.P_DDV_OPX_MDPREDICTIVO.SBN_ENVIO_COMPIN_VERSION_FARO_CON_RUT.sql
└── OPX.P_DDV_OPX_MDPREDICTIVO.SBN_LM_INPUT_DIARIO_ALFIL.sql
```

### 📄 Documentación de Desarrollo
```
old/
├── DAILY_PIPELINE.md                # Documentación del pipeline diario (incorporado en RESUMEN)
├── DEPLOYMENT_GUIDE.md              # Guía de deployment (incorporado en RESUMEN)
├── DOCKER_MANUAL_OPERACION.md       # Manual de Docker (incorporado en RESUMEN)
├── ORDEN_EJECUCION_QUERIES.md       # Orden de ejecución de queries
└── tablas_modelos.md                # Descripción de tablas del modelo
```

### 🛠️ Archivos de Desarrollo
```
old/
├── Makefile                         # Comandos make para desarrollo
├── prepare_delivery.sh              # Script de preparación de entrega
├── ft3_colmena_20250923.tar.gz     # Entrega anterior
├── action_codes_mapping.sql         # Mapeo de códigos de acción
├── query_1.sql                      # Query de análisis
├── run_monthly_pipeline.sh          # Pipeline mensual antiguo
├── run_weekly_pipeline.sh           # Pipeline semanal antiguo
├── cron/                            # Configuración de cron antiguo
├── sql/                             # Directorio SQL antiguo
└── docs/                            # Documentación de desarrollo
```

---

## DIRECTORIOS CREADOS AUTOMÁTICAMENTE

Estos directorios se crean en ejecución y **no están incluidos** en el tar.gz:

```
logs/           # Logs de ejecución del pipeline
results/        # Resultados de predicciones (Excel, CSV)
reports/        # Reportes de auditoría del modelo
data/           # Datos temporales
```

---

## ARCHIVOS EXCLUIDOS SIEMPRE

Por seguridad y buenas prácticas, estos archivos **NUNCA** se incluyen:

```
.env                    # Credenciales de Snowflake (SENSIBLE)
__pycache__/            # Cache de Python
*.pyc                   # Bytecode compilado
.DS_Store               # Metadata de macOS
Icon                    # Iconos de macOS
.claude/                # Configuración de Claude Code
```

---

## TAMAÑO DEL PAQUETE

| Item | Tamaño Aproximado |
|------|-------------------|
| **Código fuente** | < 1 MB |
| **Variables_cat_train.py** | 10 KB |
| **Modelos entrenados** | ~13 MB |
| **Docker + scripts** | < 0.5 MB |
| **Documentación** | < 0.5 MB |
| **TOTAL** | **~14 MB** |

---

## VERIFICACIÓN DE INTEGRIDAD

Para verificar que el archivo comprimido contiene todos los archivos esenciales:

```bash
# Listar contenido del tar.gz
tar -tzf ft3_colmena_production_20251002.tar.gz | head -50

# Extraer en directorio temporal para verificar
mkdir -p /tmp/ft3_verify
tar -xzf ft3_colmena_production_20251002.tar.gz -C /tmp/ft3_verify
ls -R /tmp/ft3_verify
```

### Checklist de Archivos Críticos

**Producción Diaria:**
- [ ] src/data_loader.py
- [ ] src/feature_engineering.py
- [ ] src/model_training.py
- [ ] FT3_dia.py (pipeline diario automático)
- [ ] run_daily_pipeline.py
- [ ] Variables_cat_train.py (⚠️ CRÍTICO - define las 269 variables)
- [ ] models/fasttrack_model.pkl
- [ ] models/feature_fasttrack.pkl
- [ ] docker/Dockerfile
- [ ] docker-compose.yml
- [ ] config.yaml
- [ ] requirements.txt
- [ ] query_diaria.sql
- [ ] query_lunes.sql
- [ ] query_2.sql

**Herramientas de Análisis:**
- [ ] FT30.py (⚠️ IMPORTANTE - reprocesamiento con fechas personalizadas)
- [ ] analyze_threshold_optimization.py
- [ ] optimize_with_compin.py
- [ ] show_optimal_thresholds_compin.py

**Documentación:**
- [ ] RESUMEN_EJECUTIVO_COLMENA.md
- [ ] INVENTARIO_ARCHIVOS.md

---

## INSTRUCCIONES DE DESPLIEGUE

### Opción A: Despliegue con Docker (Recomendado)

```bash
# 1. Descomprimir
tar -xzf ft3_colmena_production_20251002.tar.gz
cd python_model

# 2. Configurar credenciales
cp .env.example .env
nano .env  # Editar con credenciales de Snowflake

# 3. Levantar servicio
docker-compose up -d ft3-scheduler

# 4. Verificar logs
docker-compose logs -f ft3-scheduler
```

### Opción B: Despliegue Manual (Sin Docker)

```bash
# 1. Descomprimir
tar -xzf ft3_colmena_production_20251002.tar.gz
cd python_model

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales
cp .env.example .env
nano .env

# 5. Ejecutar pipeline
python run_daily_pipeline.py
```

---

## NOTAS IMPORTANTES

1. **Archivo .env:** Debe crearse manualmente con las credenciales de Snowflake. **NO está incluido por seguridad**.

2. **Modelos pre-entrenados:** Los archivos `.pkl` en `models/` están incluidos y listos para usar. No es necesario reentrenar a menos que se requiera actualización.

3. **Persistencia de datos:** Los directorios `logs/`, `results/`, `reports/` y `data/` se crean automáticamente en la primera ejecución.

4. **Archivos en old/:** Estos archivos están disponibles en el directorio original pero **NO** en el tar.gz de producción. Pueden ser útiles para análisis histórico o debugging.

5. **Actualización de optuna_study.db:** El archivo de optimización Optuna se incluye con el historial de 28 trials (si existe). Nuevos entrenamientos continuarán desde este punto.

---

## CONTACTO

Para consultas técnicas sobre este inventario o el despliegue:

**Desarrollador:** Andrés Vergara
**Email:** andres.vergara@maindset.cl
**Equipo:** Maindset Data Science Team

---

**Documento generado:** 2 de Octubre, 2025
**Versión del paquete:** ft3_colmena_production_20251002.tar.gz
