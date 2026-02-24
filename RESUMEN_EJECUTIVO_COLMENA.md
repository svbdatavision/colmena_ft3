# RESUMEN EJECUTIVO - FASTTRACK 3.0 (FT3)
## Sistema de Predicción Automatizada de Licencias Médicas

**Fecha:** 2 de Octubre, 2025
**Versión del Sistema:** FT3 (FastTrack 3.0)
**Modelo:** LightGBM con optimización Optuna
**Período de Entrenamiento:** 2022-01-01 a 2025-09-01 (3.7 años)

---

## 1. RESUMEN EJECUTIVO

FastTrack 3.0 es un sistema de machine learning que automatiza la evaluación de licencias médicas para determinar cuáles pueden ser auto-aprobadas sin revisión manual. El sistema utiliza un modelo LightGBM entrenado con 269 variables base que predice la probabilidad de que una licencia pueda ser aprobada automáticamente.

### Beneficios Clave
- **Reducción de costos operativos:** Automatización de licencias de bajo riesgo
- **Mejora en tiempos de respuesta:** Aprobación inmediata para casos claros
- **Optimización de recursos:** Enfoque del equipo médico en casos complejos
- **Trazabilidad completa:** Auditoría y reportes detallados de todas las decisiones

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Componentes Core

```
FT3/
├── src/                          # Módulos principales del sistema
│   ├── data_loader.py           # Conexión y carga desde Snowflake
│   ├── feature_engineering.py   # Transformación de 269 variables
│   ├── model_training.py        # Entrenamiento LightGBM + Optuna
│   └── model_auditor.py         # Auditoría y métricas
│
├── Scripts de Producción
│   ├── FT3_dia.py               # Procesamiento diario automatizado
│   ├── FT30.py                  # Procesamiento con rango de fechas personalizado
│   ├── run_daily_pipeline.py    # Orquestador del pipeline diario
│   └── main.py                  # Entrenamiento y validación del modelo
│
├── Scripts de Análisis
│   ├── analyze_threshold_optimization.py  # Análisis de umbrales
│   ├── optimize_with_compin.py           # Optimización con costos COMPIN
│   └── show_optimal_thresholds_compin.py # Visualización de umbrales
│
├── Configuración
│   ├── config.yaml              # Parámetros del modelo y Snowflake
│   ├── .env                     # Credenciales (no incluido en entregas)
│   └── requirements.txt         # Dependencias Python
│
├── Queries SQL
│   ├── query_diaria.sql         # Carga diaria de licencias (Mar-Dom)
│   ├── query_lunes.sql          # Carga especial Lunes (incluye fin de semana)
│   └── query_2.sql              # Actualización de tabla de entrenamiento
│
└── Docker
    ├── docker/Dockerfile        # Imagen del contenedor
    ├── docker/crontab           # Programación de tareas
    └── docker-compose.yml       # Orquestación de servicios
```

### 2.2 Flujo de Datos

```
SNOWFLAKE (OPX.P_DDV_OPX_MDPREDICTIVO)
    ↓
[1] Carga de Licencias Nuevas (query_diaria.sql / query_lunes.sql)
    ↓
[2] Actualización Tabla Entrenamiento (query_2.sql)
    ↓
[3] Feature Engineering (269 variables base → ~500 features transformados)
    ↓
[4] Predicción LightGBM (probabilidad 0-1 de auto-aprobación)
    ↓
[5] Sistema de Semáforo
    ├── 🟢 VERDE: P ≥ 0.94 → Auto-aprobación (alta confianza)
    ├── 🟡 AMARILLO: 0.16 ≤ P < 0.94 → Revisión manual recomendada
    └── 🔴 ROJO: P < 0.16 → Alto riesgo, revisión obligatoria
    ↓
[6] Resultados a Snowflake + Reportes Excel
```

---

## 3. VARIABLES Y FEATURES

### 3.1 Categorías de Variables (269 base)

| Categoría | Cantidad | Descripción | Ejemplos |
|-----------|----------|-------------|----------|
| **Llaves** | 7 | Identificadores únicos | RUT, N_LICENCIA |
| **Fechas** | 8 | Campos temporales con derivaciones | FECHA_RECEPCION, FECHA_EMISION |
| **Categóricas** | 21 | Códigos médicos y demográficos | CIE_GRUPO, ESPECIALIDAD_MEDICA, REGION |
| **Numéricas** | 231 | Métricas históricas y conteos | DIAS_SOLICITADOS, TASA_RECHAZO_6M |
| **Binarias** | 22 | Indicadores y flags | ES_PRIMERA_LICENCIA, TIENE_PERITAJE |
| **Texto** | 2 | Procesados con TF-IDF | LM_DIAGNOSTICO, LM_ANTECEDENTES_CLINICOS |
| **Target** | 8 | Variables objetivo (excluidas) | TARGET_FT3 |

### 3.2 Transformaciones Aplicadas

1. **Fechas:** Se derivan 4 features por fecha (_day, _month, _year, _days_since_ref)
2. **Categóricas:** Label Encoding + One-Hot Encoding
3. **Texto:** TF-IDF Vectorization (n-grams 1-2)
4. **Numéricas:** Escalado y normalización
5. **Binarias:** Conversión a 0/1

**Resultado:** ~500 features finales para el modelo

---

## 4. MODELO LIGHTGBM

### 4.1 Configuración del Modelo

```yaml
Algoritmo: LightGBM Classifier
Optimización: Optuna (40 trials, 5-fold CV)
Métrica Principal: AUC-ROC
Métrica de Validación: AUC, Precision, Recall, F1-Score

Parámetros Clave:
  - scale_pos_weight: 5.0  # Penaliza falsos positivos
  - early_stopping_rounds: 50
  - eval_metric: auc
  - boosting_type: gbdt
```

### 4.2 Ventana de Entrenamiento

- **Período:** 2022-01-01 a 2025-09-01
- **Tabla Snowflake:** `MODELO_LM_202507_TRAIN`
- **Criterio de corte:** `FECHA_RECEPCION`
- **Exclusiones:**
  - Casos postnatales (CIE_GRUPO: PARTO, PUERPERIO)
  - Registros con TARGET_FT3 = NULL
  - Registros sin CIE_GRUPO

### 4.3 Definición del Target (TARGET_FT3)

**TARGET_FT3 = 1 (Auto-aprobable)** si cumple TODO lo siguiente:
- Sin observaciones médicas
- Sin peritaje médico
- Días autorizados = Días solicitados
- Sin ajustes ni modificaciones

**TARGET_FT3 = 0 (Revisión manual)** en cualquier otro caso

### 4.4 Estrategia de Validación

- **Test set:** 20% de datos (split estratificado, random_state=42)
- **Validation set:** 10% del training set (para early stopping)
- **Cross-validation:** 5-fold durante hyperparameter tuning

---

## 5. SISTEMA DE SEMÁFORO

### 5.1 Umbrales de Decisión

El sistema clasifica cada licencia en 3 categorías:

| Semáforo | Umbral | Decisión | Justificación |
|----------|--------|----------|---------------|
| 🟢 **VERDE** | P ≥ 0.94 | Auto-aprobación | Alta confianza (94%+), riesgo mínimo |
| 🟡 **AMARILLO** | 0.16 ≤ P < 0.94 | Revisión manual | Confianza media, requiere validación |
| 🔴 **ROJO** | P < 0.16 | Rechazar/Revisar | Baja confianza (<16%), revisión obligatoria |

### 5.2 Umbrales por Defecto vs Optimizados

**Umbrales por Defecto (configurados actualmente):**
- 🟢 Verde: ≥ **0.94** (94%)
- 🟡 Amarillo: **0.16** - **0.94** (16% - 94%)
- 🔴 Rojo: < **0.16** (16%)

**Justificación:**
- El umbral alto (0.94) **minimiza falsos positivos** (aprobar incorrectamente)
- El sistema prioriza **conservadorismo** para evitar aprobaciones incorrectas
- Estos umbrales pueden **optimizarse** según costos reales

### 5.3 Optimización de Umbrales

Los umbrales pueden optimizarse considerando:

```python
Costos Operacionales:
  - Costo Falso Positivo: $59,000/día (aprobar incorrectamente)
  - Costo Falso Negativo: $20,000 (rechazar incorrectamente)
  - Costo Revisión Manual: $5,000 por caso
  - Tasa Reversión COMPIN: 30% (configurable)
```

**Scripts de optimización:**
- `optimize_with_compin.py` - Encuentra umbrales óptimos basados en costos
- `show_optimal_thresholds_compin.py` - Visualiza recomendaciones
- `analyze_threshold_optimization.py` - Análisis detallado de sensibilidad

**Nota:** El sistema permite ajustar estos umbrales sin reentrenar el modelo, solo modificando los parámetros de clasificación.

---

## 6. PIPELINE DIARIO DE PRODUCCIÓN

### 6.1 Programación Automática (Docker Cron)

```bash
# Todos los días a las 6:00 AM (Chile)
0 6 * * * /app/run_daily_pipeline.sh
```

### 6.2 Flujo del Pipeline Diario

**Lunes (incluye fin de semana):**
```
1a. Ejecutar query_lunes.sql       → Cargar licencias Sab-Dom
1b. Ejecutar query_diaria.sql      → Cargar licencias Lun
2.  Ejecutar query_2.sql           → Actualizar MODELO_LM_202507_TRAIN
3.  Ejecutar FT3_dia.py            → Generar predicciones
4.  Guardar resultados en Snowflake + Excel
```

**Martes a Domingo:**
```
1. Ejecutar query_diaria.sql       → Cargar licencias del día anterior
2. Ejecutar query_2.sql            → Actualizar MODELO_LM_202507_TRAIN
3. Ejecutar FT3_dia.py             → Generar predicciones
4. Guardar resultados en Snowflake + Excel
```

### 6.3 Outputs del Pipeline

1. **Snowflake:** Tabla `FT30_PREDICCIONES_DIARIAS` con:
   - N_LICENCIA
   - PROBABILIDAD_APROBACION
   - SEMAFORO (VERDE/AMARILLO/ROJO)
   - FECHA_PREDICCION
   - VERSION_MODELO

2. **Excel:** Reportes en `results/` con:
   - Distribución de semáforos
   - Estadísticas descriptivas
   - Casos de alto riesgo destacados

3. **Logs:** Auditoría completa en `logs/`

---

## 7. DESPLIEGUE CON DOCKER

### 7.1 Contenedores Disponibles

#### **ft3-scheduler** (Producción Diaria)
```bash
# Levantar servicio programado
docker-compose up -d ft3-scheduler

# Características:
# - Cron integrado (6:00 AM diario)
# - Persistencia de modelos y resultados
# - Logs en tiempo real
# - Reconexión automática a Snowflake
```

#### **ft3-training** (Reentrenamiento)
```bash
# Ejecutar reentrenamiento manual
docker-compose run ft3-training

# Características:
# - Hyperparameter tuning con Optuna (40 trials)
# - Validación cruzada 5-fold
# - Guardado de progreso (optuna_study.db)
# - Puede retomarse si se interrumpe
```

### 7.2 Configuración de Variables de Entorno

Crear archivo `.env` en `python_model/`:

```bash
# Snowflake Connection
SF_USER=tu_usuario
SF_PASSWORD=tu_password
SF_ACCOUNT=COLMENA-ISAPRE_COLMENA
SF_WAREHOUSE=P_ML
SF_DATABASE=OPX
SF_SCHEMA=P_DDV_OPX_MDPREDICTIVO
SF_ROLE=EX_ML

# Timezone
TZ=America/Santiago
```

### 7.3 Volúmenes Persistentes

```yaml
Volúmenes montados:
  - ./models     → Modelos entrenados (.pkl)
  - ./results    → Reportes Excel y CSV
  - ./reports    → Auditorías del modelo
  - ./logs       → Logs de ejecución
  - ./data       → Datos temporales
```

---

## 8. MANTENIMIENTO Y MONITOREO

### 8.1 Tareas Programadas

| Frecuencia | Tarea | Script | Descripción |
|------------|-------|--------|-------------|
| **Diaria** (6 AM) | Predicciones | `run_daily_pipeline.py` | Procesa licencias nuevas |
| **Semanal** (Lunes 7 AM) | Validación | `scripts/validate_model.py` | Verifica performance del modelo |
| **Mensual** (Día 1, 3 AM) | Update Variables | `scripts/update_variables.py` | Actualiza features |
| **Trimestral** (2 AM) | Reentrenamiento | `scripts/retrain_model.py` | Reentrena modelo completo |

### 8.2 Métricas de Monitoreo

**Archivo:** `src/model_auditor.py` genera reportes con:

- **Performance Metrics:** AUC-ROC, Precision, Recall, F1-Score
- **Confusion Matrix:** Distribución de TP, TN, FP, FN
- **Feature Importance:** Top 50 variables más influyentes
- **Drift Detection:** Comparación con baseline histórico
- **Distribution Analysis:** Cambios en distribución de probabilidades

**Reportes guardados en:** `reports/audit_YYYYMMDD_HHMMSS.json`

---

## 9. ARCHIVOS ESENCIALES PARA PRODUCCIÓN

### 9.1 Archivos de Código

```
✅ ESENCIALES (incluidos en entrega Docker)
├── src/data_loader.py
├── src/feature_engineering.py
├── src/model_training.py
├── src/model_auditor.py
├── FT3_dia.py
├── run_daily_pipeline.py
├── main.py
├── config.yaml
├── requirements.txt
├── docker/Dockerfile
├── docker/crontab
├── docker-compose.yml
├── query_diaria.sql
├── query_lunes.sql
└── query_2.sql
```

### 9.2 Modelos Pre-entrenados

```
✅ INCLUIDOS (directorio models/)
├── fasttrack_model.pkl          # Modelo LightGBM entrenado
├── feature_fasttrack.pkl        # Transformers (TF-IDF, encoders)
├── optuna_study.db              # Historial de optimización
└── fasttrack_model_feature_importance.csv
```

### 9.3 Archivos Movidos a old/ (No esenciales)

```
❌ ARCHIVOS DE ANÁLISIS Y DESARROLLO (movidos a old/)
├── FT20.py                      # Versión anterior del modelo
├── FT30.py                      # Versión standalone (reemplazada por FT3_dia.py)
├── histogramas.py               # Análisis exploratorio
├── analyze_threshold_optimization.py
├── optimize_with_compin.py      # Optimización de umbrales (opcional)
├── show_optimal_thresholds_compin.py
├── check_fields_simple.py
├── compare_queries.py
├── analisis_histogramas/        # Análisis histórico
├── OPX.P_DDV_OPX_MDPREDICTIVO.*.sql  # Definiciones de tablas Snowflake
└── Otros archivos de análisis
```

---

## 10. INSTRUCCIONES DE DESPLIEGUE

### 10.1 Primera Instalación

```bash
# 1. Descomprimir entrega
tar -xzf ft3_colmena_production.tar.gz
cd python_model

# 2. Configurar credenciales
cp .env.example .env
# Editar .env con credenciales de Snowflake

# 3. Verificar que existan los modelos
ls -lh models/
# Debe mostrar: fasttrack_model.pkl, feature_fasttrack.pkl

# 4. Levantar contenedor de producción
docker-compose up -d ft3-scheduler

# 5. Verificar logs
docker-compose logs -f ft3-scheduler
```

### 10.2 Ejecución Manual (Sin Docker)

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar .env
cp .env.example .env
# Editar .env

# 4. Ejecutar pipeline diario
python run_daily_pipeline.py

# 5. Reentrenar modelo (opcional)
python main.py --mode train
```

### 10.3 Reentrenamiento del Modelo

```bash
# Opción A: Con Docker (recomendado)
docker-compose run ft3-training

# Opción B: Sin Docker
python main.py --mode train

# Características:
# - Duración: ~3-4 horas (40 trials Optuna)
# - Progreso guardado en models/optuna_study.db
# - Puede retomarse si se interrumpe
# - Genera nuevo fasttrack_model.pkl
```

---

## 11. TROUBLESHOOTING

### 11.1 Problemas Comunes

#### Error: "Loaded 0 rows"
**Causa:** La fecha de corte en FT30.py no tiene datos
**Solución:** Usar fechas históricas con datos reales

#### Error: "idf vector is not fitted"
**Causa:** Incompatibilidad de versión de scikit-learn
**Solución:**
```bash
pip install --upgrade scikit-learn>=1.3.0
python fix_transformers.py  # Regenera transformers
```

#### Error: "Feature names seen at fit time, yet now missing"
**Causa:** Licencias pendientes sin columnas de fecha
**Solución:** Ya manejado automáticamente en `feature_engineering.py:83-118`

### 11.2 Logs y Auditoría

```bash
# Ver logs del contenedor
docker-compose logs -f ft3-scheduler

# Ver logs locales
tail -f logs/ft3_YYYYMMDD.log

# Ver último reporte de auditoría
ls -lt reports/ | head -1
```

### 11.3 Contacto de Soporte

Para soporte técnico, contactar a:
- **Desarrollador:** Andrés Vergara (andres.vergara@maindset.cl)
- **Equipo:** Maindset Data Science Team

---

## 12. RESUMEN DE CAMBIOS VS FT2.0

| Aspecto | FT2.0 | FT3.0 |
|---------|-------|-------|
| **Algoritmo** | LightGBM básico | LightGBM + Optuna (40 trials) |
| **Features** | ~180 variables | 269 variables base → ~500 transformadas |
| **Pipeline** | Scripts separados | Pipeline integrado con Docker |
| **Monitoreo** | Manual | Auditoría automática |
| **Deployment** | Local | Docker con cron programado |
| **Optimización** | Fija | Retomable con optuna_study.db |
| **Umbrales** | Estáticos | Optimizables por costos |
| **Texto** | No procesado | TF-IDF en diagnósticos |

---

## 13. PRÓXIMOS PASOS RECOMENDADOS

1. **Validación Inicial (Semana 1)**
   - Ejecutar pipeline manualmente y validar resultados
   - Comparar predicciones con decisiones manuales históricas
   - Ajustar umbrales si es necesario

2. **Piloto Controlado (Mes 1)**
   - Ejecutar en paralelo con proceso manual
   - Monitorear tasa de aciertos en zona VERDE
   - Documentar casos problemáticos

3. **Despliegue Gradual (Mes 2-3)**
   - Comenzar auto-aprobando solo zona VERDE con P ≥ 0.94
   - Expandir gradualmente según confianza (ajustar umbrales si es necesario)
   - Reentrenar modelo mensualmente

4. **Optimización Continua (Mes 4+)**
   - Ajustar umbrales con datos reales de costos
   - Incorporar feedback de equipos médicos
   - Explorar nuevas variables predictivas

---

**Documento generado:** 2 de Octubre, 2025
**Versión:** 1.0
**Autor:** Sistema FT3 - Maindset para Isapre Colmena
