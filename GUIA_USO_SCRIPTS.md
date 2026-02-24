# GUÍA DE USO DE SCRIPTS - FASTTRACK 3.0

**Fecha:** 2 de Octubre, 2025

---

## 1. SCRIPTS DE PRODUCCIÓN

### 1.1 FT3_dia.py - Pipeline Diario Automatizado

**Propósito:** Procesamiento automático diario de licencias nuevas

**Cuándo se usa:**
- Ejecutado automáticamente por Docker a las 6:00 AM
- Procesa licencias del día anterior
- Los lunes procesa todo el fin de semana

**Ejecución:**
```bash
# Ejecutado automáticamente por run_daily_pipeline.py
# NO requiere parámetros
python run_daily_pipeline.py
```

**Output:**
- Predicciones en Snowflake: `FT30_PREDICCIONES_DIARIAS`
- Reporte Excel en: `results/predicciones_YYYYMMDD.xlsx`
- Logs en: `logs/ft3_YYYYMMDD.log`

---

### 1.2 FT30.py - Reprocesamiento con Fechas Personalizadas

**Propósito:** Procesar licencias para períodos específicos

**Casos de uso:**

#### ✅ Caso 1: Comparar Resultados Después de Reentrenamiento
```bash
# 1. Guardar predicciones anteriores
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 > resultados_modelo_anterior.txt

# 2. Reentrenar modelo
python main.py --mode train

# 3. Reprocesar mismo período
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 > resultados_modelo_nuevo.txt

# 4. Comparar diferencias
diff resultados_modelo_anterior.txt resultados_modelo_nuevo.txt
```

#### ✅ Caso 2: Recuperar Datos de Caída del Sistema
```bash
# Si el sistema estuvo caído del 10 al 15 de septiembre
python FT30.py --desde 2024-09-10 --hasta 2024-09-15
```

#### ✅ Caso 3: Análisis Retrospectivo de Performance
```bash
# Analizar últimos 90 días
python FT30.py --ultimos-dias 90

# Analizar un trimestre completo
python FT30.py --desde 2024-07-01 --hasta 2024-09-30
```

#### ✅ Caso 4: Validación de Umbrales
```bash
# Probar con optimización de umbrales
python FT30.py --desde 2024-08-01 --hasta 2024-08-31

# Probar sin optimización (umbrales fijos)
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 --no-optimizar

# Comparar resultados
```

#### ✅ Caso 5: Análisis de Sensibilidad de Costos
```bash
# Escenario base
python FT30.py --desde 2024-08-01 --hasta 2024-08-31

# Escenario optimista (alta reversión COMPIN)
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 --compin 0.5

# Escenario pesimista (baja reversión COMPIN)
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 --compin 0.1
```

**Parámetros disponibles:**
```bash
--desde YYYY-MM-DD          # Fecha inicio del rango
--hasta YYYY-MM-DD          # Fecha fin del rango
--ultimos-dias N            # Procesar últimos N días (sobrescribe --desde)
--no-optimizar              # Usar umbrales fijos (no optimizar)
--compin FLOAT              # Tasa de reversión COMPIN (0.0 a 1.0)
--costo-manual INT          # Costo de revisión manual ($)
```

**⚠️ IMPORTANTE:**
- Usar **fechas históricas** (no futuras)
- La fecha de corte es `FECHA_EMISION_DT < fecha_hasta`
- Si no hay datos para el rango, mostrará "Loaded 0 rows"

---

### 1.3 main.py - Entrenamiento del Modelo

**Propósito:** Entrenar o reentrenar el modelo LightGBM

**Casos de uso:**

#### Entrenamiento Completo (Recomendado)
```bash
# Con hyperparameter tuning (40 trials Optuna)
# Duración: ~3-4 horas
python main.py --mode train
```

#### Entrenamiento Rápido
```bash
# Sin tuning (usa parámetros de config.yaml)
# Duración: ~30 minutos
python main.py --mode train --no-tuning
```

#### Predicción en Licencias Específicas
```bash
# Predecir licencias individuales
python main.py --mode predict --ids LIC001 LIC002 LIC003
```

**Output:**
- Modelo guardado en: `models/fasttrack_model.pkl`
- Transformers en: `models/feature_fasttrack.pkl`
- Feature importance: `models/fasttrack_model_feature_importance.csv`
- Resultados: `results/training_results_YYYYMMDD_HHMMSS.csv`

---

## 2. SCRIPTS DE ANÁLISIS

### 2.1 optimize_with_compin.py - Optimización de Umbrales

**Propósito:** Encontrar umbrales óptimos considerando costos operacionales

**Umbrales actuales por defecto:**
- 🟢 Verde: ≥ 0.94 (94%)
- 🟡 Amarillo: 0.16 - 0.94 (16% - 94%)
- 🔴 Rojo: < 0.16 (16%)

**Cuándo usar:**
- Cuando cambian los costos operacionales
- Para ajustar la agresividad del modelo (más/menos conservador)
- Análisis de sensibilidad
- Evaluar impacto de cambios en política COMPIN

**Ejecución:**
```bash
python optimize_with_compin.py
```

**Output:**
- Umbrales óptimos recomendados para Verde/Amarillo/Rojo
- Análisis de costos por escenario
- Gráficos de distribución
- Comparación con umbrales actuales

---

### 2.2 show_optimal_thresholds_compin.py - Visualización de Umbrales

**Propósito:** Mostrar recomendaciones de umbrales de forma visual

**Ejecución:**
```bash
python show_optimal_thresholds_compin.py
```

**Output:**
- Tabla de umbrales recomendados
- Distribución esperada de casos (Verde/Amarillo/Rojo)
- Estimación de costos

---

### 2.3 analyze_threshold_optimization.py - Análisis Detallado

**Propósito:** Análisis profundo del comportamiento de umbrales

**Cuándo usar:**
- Para entender por qué se recomiendan ciertos umbrales
- Análisis de sensibilidad detallado
- Debugging de optimización

**Ejecución:**
```bash
python analyze_threshold_optimization.py
```

**Output:**
- Gráficos de costo vs umbral
- Análisis de trade-offs
- Estadísticas descriptivas

---

## 3. FLUJOS DE TRABAJO COMUNES

### 3.1 Workflow: Reentrenamiento Trimestral

```bash
# Paso 1: Backup del modelo anterior
cp models/fasttrack_model.pkl models/fasttrack_model_backup_$(date +%Y%m%d).pkl

# Paso 2: Generar predicciones con modelo anterior (para comparación)
python FT30.py --ultimos-dias 90 > baseline_anterior.txt

# Paso 3: Reentrenar modelo
python main.py --mode train

# Paso 4: Generar predicciones con modelo nuevo
python FT30.py --ultimos-dias 90 > baseline_nuevo.txt

# Paso 5: Comparar performance
diff baseline_anterior.txt baseline_nuevo.txt

# Paso 6: Optimizar umbrales
python optimize_with_compin.py

# Paso 7: Si mejora, mantener. Si no, revertir:
# cp models/fasttrack_model_backup_YYYYMMDD.pkl models/fasttrack_model.pkl
```

---

### 3.2 Workflow: Análisis Mensual de Performance

```bash
# Paso 1: Procesar último mes
python FT30.py --ultimos-dias 30

# Paso 2: Revisar umbrales óptimos
python show_optimal_thresholds_compin.py

# Paso 3: Análisis detallado
python analyze_threshold_optimization.py

# Paso 4: Generar reportes
# (revisar results/ y reports/)
```

---

### 3.3 Workflow: Recuperación de Incidente

```bash
# Escenario: Sistema caído del 15 al 20 de septiembre

# Paso 1: Verificar conectividad Snowflake
python -c "from src.data_loader import SnowflakeDataLoader; loader = SnowflakeDataLoader(); loader.connect()"

# Paso 2: Reprocesar período perdido
python FT30.py --desde 2024-09-15 --hasta 2024-09-20

# Paso 3: Verificar que las predicciones se guardaron en Snowflake
# (revisar tabla FT30_PREDICCIONES_DIARIAS)

# Paso 4: Reiniciar pipeline diario
docker-compose restart ft3-scheduler
```

---

### 3.4 Workflow: Ajuste de Umbrales por Cambio de Costos

```bash
# Escenario: El costo de revisión manual cambió de $5,000 a $7,000

# Paso 1: Editar config.yaml
# (actualizar parámetro cost_manual_review)

# Paso 2: Re-optimizar umbrales
python optimize_with_compin.py --costo-manual 7000

# Paso 3: Probar en período histórico
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 --costo-manual 7000

# Paso 4: Comparar con umbrales anteriores
python FT30.py --desde 2024-08-01 --hasta 2024-08-31 --no-optimizar
```

---

## 4. TROUBLESHOOTING

### Error: "Loaded 0 rows"

**Causa:** No hay datos para el rango de fechas especificado

**Solución:**
```bash
# Verificar fechas disponibles en Snowflake
# Usar fechas históricas, no futuras
python FT30.py --desde 2024-06-01 --hasta 2024-08-31
```

---

### Error: "Feature names mismatch"

**Causa:** El modelo fue entrenado con features diferentes

**Solución:**
```bash
# Opción 1: Regenerar transformers
python fix_transformers.py

# Opción 2: Reentrenar modelo completo
python main.py --mode train
```

---

### Warning: "Using all features (no filtering)"

**Esto es NORMAL:** El modelo ahora usa todas las 269 variables base (sin filtrado por IV)

---

## 5. MEJORES PRÁCTICAS

### ✅ DO (Hacer)

1. **Usar FT3_dia.py para pipeline diario** (automático vía Docker)
2. **Usar FT30.py para análisis ad-hoc** (reprocesamiento, comparaciones)
3. **Hacer backup antes de reentrenar**
4. **Probar en datos históricos antes de producción**
5. **Documentar cambios de umbrales**

### ❌ DON'T (No Hacer)

1. **No usar FT30.py con fechas futuras** (no hay datos)
2. **No reentrenar en producción sin validación**
3. **No cambiar umbrales sin análisis**
4. **No ejecutar FT3_dia.py y FT30.py simultáneamente** (conflictos de Snowflake)
5. **No olvidar Variables_cat_train.py** (archivo crítico)

---

## 6. RESUMEN RÁPIDO

| Script | Cuándo Usar | Frecuencia | Automático |
|--------|-------------|------------|------------|
| **FT3_dia.py** | Pipeline diario | Diaria (6 AM) | ✅ Sí (Docker) |
| **FT30.py** | Reprocesamiento | Ad-hoc | ❌ No |
| **main.py** | Entrenamiento | Trimestral | ❌ No |
| **optimize_with_compin.py** | Ajustar umbrales | Mensual | ❌ No |
| **show_optimal_thresholds_compin.py** | Visualizar umbrales | Ad-hoc | ❌ No |
| **analyze_threshold_optimization.py** | Análisis profundo | Ad-hoc | ❌ No |

---

**Documento generado:** 2 de Octubre, 2025
**Autor:** Sistema FT3 - Maindset para Isapre Colmena
