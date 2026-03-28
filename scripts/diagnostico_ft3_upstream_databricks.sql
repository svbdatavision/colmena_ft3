-- ============================================================================
-- FT3 | Diagnóstico upstream end-to-end (Databricks SQL)
-- Objetivo:
--   Detectar en qué etapa se pierde volumen entre INPUT -> BASE -> OPTIMIZADO -> TRAIN
--   y dejar evidencia trazable para ticket con equipo upstream.
--
-- Esquema objetivo:
--   OPX.P_DDV_OPX_MDPREDICTIVO
--
-- Ventana de análisis:
--   últimos 7 días (hardcoded para ejecución simple).
-- ============================================================================

-- ============================================================================
-- 0) Contexto de sesión (evidencia de catalog/schema activos)
-- ============================================================================
SELECT
  current_catalog() AS current_catalog,
  current_schema() AS current_schema,
  current_user() AS current_user,
  current_timestamp() AS executed_at;

-- ============================================================================
-- 1) Existencia de tablas clave
-- ============================================================================
SHOW TABLES IN OPX.P_DDV_OPX_MDPREDICTIVO LIKE 'SBN_LM_INPUT_DIARIO_ALFIL';
SHOW TABLES IN OPX.P_DDV_OPX_MDPREDICTIVO LIKE 'MODELO_LM_202507_BASE';
SHOW TABLES IN OPX.P_DDV_OPX_MDPREDICTIVO LIKE 'MODELO_LM_202507_OPTIMIZADO';
SHOW TABLES IN OPX.P_DDV_OPX_MDPREDICTIVO LIKE 'MODELO_LM_202507_TRAIN';

-- ============================================================================
-- 2) Inventario de columnas (útil para validar nombres de fecha en origen)
-- ============================================================================
DESCRIBE TABLE OPX.P_DDV_OPX_MDPREDICTIVO.SBN_LM_INPUT_DIARIO_ALFIL;
DESCRIBE TABLE OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE;
DESCRIBE TABLE OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO;
DESCRIBE TABLE OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN;

-- ============================================================================
-- 3) Conteo total por etapa (termómetro general)
-- ============================================================================
WITH stage_counts AS (
  SELECT 'INPUT_DIARIO_ALFIL' AS etapa, COUNT(*) AS total_filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.SBN_LM_INPUT_DIARIO_ALFIL
  UNION ALL
  SELECT 'MODELO_BASE' AS etapa, COUNT(*) AS total_filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE
  UNION ALL
  SELECT 'MODELO_OPTIMIZADO' AS etapa, COUNT(*) AS total_filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO
  UNION ALL
  SELECT 'MODELO_TRAIN' AS etapa, COUNT(*) AS total_filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
)
SELECT *
FROM stage_counts
ORDER BY etapa;

-- ============================================================================
-- 4) Movimiento últimos 7 días en BASE/OPTIMIZADO/TRAIN
-- ============================================================================
WITH recent_flow AS (
  SELECT
    'MODELO_BASE' AS etapa,
    DATE(FECHA_RECEPCION) AS fecha_recepcion,
    COUNT(*) AS filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE
  WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -7)
  GROUP BY DATE(FECHA_RECEPCION)

  UNION ALL

  SELECT
    'MODELO_OPTIMIZADO' AS etapa,
    DATE(FECHA_RECEPCION) AS fecha_recepcion,
    COUNT(*) AS filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO
  WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -7)
  GROUP BY DATE(FECHA_RECEPCION)

  UNION ALL

  SELECT
    'MODELO_TRAIN' AS etapa,
    DATE(FECHA_RECEPCION) AS fecha_recepcion,
    COUNT(*) AS filas
  FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
  WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -7)
  GROUP BY DATE(FECHA_RECEPCION)
)
SELECT *
FROM recent_flow
ORDER BY fecha_recepcion DESC, etapa;

-- ============================================================================
-- 5) Calidad de datos en etapas intermedias y salida
-- ============================================================================
SELECT
  'MODELO_BASE' AS etapa,
  COUNT(*) AS total,
  COALESCE(SUM(CASE WHEN FECHA_EMISION_DT IS NULL THEN 1 ELSE 0 END), 0) AS nulos_fecha_emision_dt,
  COALESCE(SUM(CASE WHEN CIE_GRUPO IS NULL OR TRIM(CIE_GRUPO) = '' THEN 1 ELSE 0 END), 0) AS nulos_cie_grupo,
  COALESCE(SUM(CASE WHEN CIE_GRUPO IN ('PARTO', 'PUERPERIO') THEN 1 ELSE 0 END), 0) AS parto_puerperio
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE
UNION ALL
SELECT
  'MODELO_OPTIMIZADO' AS etapa,
  COUNT(*) AS total,
  COALESCE(SUM(CASE WHEN FECHA_EMISION_DT IS NULL THEN 1 ELSE 0 END), 0) AS nulos_fecha_emision_dt,
  COALESCE(SUM(CASE WHEN CIE_GRUPO IS NULL OR TRIM(CIE_GRUPO) = '' THEN 1 ELSE 0 END), 0) AS nulos_cie_grupo,
  COALESCE(SUM(CASE WHEN CIE_GRUPO IN ('PARTO', 'PUERPERIO') THEN 1 ELSE 0 END), 0) AS parto_puerperio
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO
UNION ALL
SELECT
  'MODELO_TRAIN' AS etapa,
  COUNT(*) AS total,
  COALESCE(SUM(CASE WHEN FECHA_EMISION_DT IS NULL THEN 1 ELSE 0 END), 0) AS nulos_fecha_emision_dt,
  COALESCE(SUM(CASE WHEN CIE_GRUPO IS NULL OR TRIM(CIE_GRUPO) = '' THEN 1 ELSE 0 END), 0) AS nulos_cie_grupo,
  COALESCE(SUM(CASE WHEN CIE_GRUPO IN ('PARTO', 'PUERPERIO') THEN 1 ELSE 0 END), 0) AS parto_puerperio
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN;

-- ============================================================================
-- 6) Diagnóstico puntual de caída en OPTIMIZADO (tu query 4 mejorada)
-- ============================================================================
SELECT
  COUNT(*) AS total_opt,
  COALESCE(SUM(CASE WHEN FECHA_EMISION_DT IS NOT NULL THEN 1 ELSE 0 END), 0) AS pasan_filtro_emision_dt,
  COALESCE(SUM(CASE WHEN FECHA_EMISION_DT IS NULL THEN 1 ELSE 0 END), 0) AS nulos_emision_dt,
  COALESCE(SUM(CASE WHEN CIE_GRUPO NOT IN ('PARTO', 'PUERPERIO') THEN 1 ELSE 0 END), 0) AS pasan_filtro_cie,
  COALESCE(SUM(CASE WHEN TARGET_APRUEBA IS NULL THEN 1 ELSE 0 END), 0) AS pendientes_sin_dictamen
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO;

-- ============================================================================
-- 7) Muestras para inspección rápida (si hay datos)
-- ============================================================================
SELECT *
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE
WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -7)
LIMIT 20;

SELECT *
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO
WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -7)
LIMIT 20;

SELECT *
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -7)
LIMIT 20;

-- ============================================================================
-- 8) Persistencia opcional a tabla de auditoría (para adjuntar en ticket)
--    Si no querés persistir, podés omitir desde CREATE TABLE hacia abajo.
-- ============================================================================
CREATE TABLE IF NOT EXISTS OPX.P_DDV_OPX_MDPREDICTIVO.FT3_DIAGNOSTICO_AUDITORIA (
  RUN_TS TIMESTAMP,
  ETAPA STRING,
  METRICA STRING,
  VALOR BIGINT,
  DETALLE STRING
);

INSERT INTO OPX.P_DDV_OPX_MDPREDICTIVO.FT3_DIAGNOSTICO_AUDITORIA
SELECT current_timestamp(), 'INPUT_DIARIO_ALFIL', 'row_count', COUNT(*), 'Conteo total tabla origen'
FROM OPX.P_DDV_OPX_MDPREDICTIVO.SBN_LM_INPUT_DIARIO_ALFIL
UNION ALL
SELECT current_timestamp(), 'MODELO_BASE', 'row_count', COUNT(*), 'Conteo total tabla base'
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE
UNION ALL
SELECT current_timestamp(), 'MODELO_OPTIMIZADO', 'row_count', COUNT(*), 'Conteo total tabla optimizada'
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO
UNION ALL
SELECT current_timestamp(), 'MODELO_TRAIN', 'row_count', COUNT(*), 'Conteo total tabla train'
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
UNION ALL
SELECT current_timestamp(), 'MODELO_OPTIMIZADO', 'fe_emision_dt_null', COALESCE(SUM(CASE WHEN FECHA_EMISION_DT IS NULL THEN 1 ELSE 0 END), 0), 'Nulos FECHA_EMISION_DT'
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_OPTIMIZADO
UNION ALL
SELECT current_timestamp(), 'MODELO_TRAIN', 'target_aprueba_null', COALESCE(SUM(CASE WHEN TARGET_APRUEBA IS NULL THEN 1 ELSE 0 END), 0), 'Pendientes sin dictamen'
FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN;

-- Ver lo recién insertado (última corrida)
SELECT *
FROM OPX.P_DDV_OPX_MDPREDICTIVO.FT3_DIAGNOSTICO_AUDITORIA
WHERE RUN_TS >= date_add(current_timestamp(), -1)
ORDER BY RUN_TS DESC, ETAPA, METRICA;
