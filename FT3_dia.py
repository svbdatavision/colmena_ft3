"""
FT3_dia.py - Script para procesamiento DIARIO del modelo FastTrack 3.0

Este script está diseñado para ejecutarse diariamente y procesar licencias médicas
usando la tabla MODELO_LM_202507_TRAIN.

FLUJO DE DATOS:
1. Input: OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
2. Procesamiento: 
   - Sin parámetros: Solo licencias sin dictamen (TARGET_APRUEBA IS NULL)
   - Con fechas: Todas las licencias del período especificado
3. Modelo: Aplica FastTrack 3.0 para predicciones
4. Output: Resultados en Databricks SQL (FT30_PREDICCIONES_DIARIAS y FT30_LICENCIAS_PENDIENTES)

ACTUALIZACIÓN 2025-09-05:
- SIEMPRE usa MODELO_LM_202507_TRAIN (no MODELO_LM_DIARIO_TRAIN)
- Sin parámetros: procesa solo licencias pendientes (sin dictamen)
- Con parámetros de fecha: procesa todas las licencias del período
- Solo guarda resultados en Databricks SQL, no genera Excel por defecto
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('src')
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.data_loader import SnowflakeDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import LightGBMTrainer
import joblib


def _write_dataframe_to_table(conn, df, table_name, batch_size=200, **_kwargs):
    """
    Reemplazo compatible de write_pandas usando executemany sobre DB-API.
    """
    if df is None or len(df) == 0:
        return True, 0, 0, None

    columns = [str(col).strip() for col in df.columns]
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    cursor = conn.cursor()
    nchunks = 0
    nrows = 0
    try:
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start:start + batch_size]
            records = []
            for row in batch_df.itertuples(index=False, name=None):
                normalized = []
                for value in row:
                    if pd.isna(value):
                        normalized.append(None)
                    elif isinstance(value, pd.Timestamp):
                        normalized.append(value.to_pydatetime())
                    elif isinstance(value, np.generic):
                        normalized.append(value.item())
                    else:
                        normalized.append(value)
                records.append(tuple(normalized))

            if records:
                cursor.executemany(insert_sql, records)
                nchunks += 1
                nrows += len(records)

        return True, nchunks, nrows, None
    except Exception:
        raise
    finally:
        cursor.close()


def align_features_with_model(X, model_path="models/fasttrack_model.pkl"):
    """
    Aligns the features of the transformed data with what the model expects.
    This handles cases where feature engineering produces a different number of features.
    """
    # Load model to get expected features
    model_data = joblib.load(model_path)
    expected_features = model_data.get('feature_names', None)
    
    if expected_features is None:
        # If no feature names stored, just return as is
        return X
    
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Get current features
    current_features = X.columns.tolist()
    
    # Find missing and extra features
    missing_features = set(expected_features) - set(current_features)
    extra_features = set(current_features) - set(expected_features)
    
    if missing_features:
        print(f"   ⚠ Missing {len(missing_features)} features, adding with zeros")
        for feat in missing_features:
            X[feat] = 0
    
    if extra_features:
        print(f"   ⚠ Removing {len(extra_features)} extra features")
        # Mostrar TODAS las features extras para diagnóstico completo
        print(f"      Features extras encontradas:")
        for i, feat in enumerate(sorted(list(extra_features)), 1):
            print(f"        {i:2d}. {feat}")
        X = X.drop(columns=list(extra_features))
    
    # Reorder columns to match expected order
    X = X[expected_features]
    
    return X

def optimize_thresholds_by_cost(probabilities, true_labels, dias_solicitados, 
                               cost_fp_per_day=59000, cost_fn=20000, cost_manual_review=5000,
                               compin_reversion_rate=0.0):
    """
    Optimiza los umbrales del semáforo basándose en los costos de errores.
    
    Parámetros:
    - probabilities: probabilidades predichas por el modelo
    - true_labels: etiquetas verdaderas (1=aprobado, 0=rechazado)
    - dias_solicitados: días solicitados para cada licencia
    - cost_fp_per_day: costo por día de falso positivo (aprobar incorrectamente)
    - cost_fn: costo de falso negativo (rechazar incorrectamente)
    - cost_manual_review: costo de revisión manual por caso en zona amarilla
    - compin_reversion_rate: tasa de reversión del COMPIN (0.0 a 1.0)
    
    Retorna:
    - optimal_threshold_verde: umbral óptimo para zona verde
    - optimal_threshold_amarillo: umbral óptimo para zona amarilla
    - cost_analysis: análisis detallado de costos
    
    Nota sobre COMPIN:
    Si compin_reversion_rate > 0, el costo efectivo de un FP se ajusta considerando
    que un porcentaje de rechazos serán revertidos por COMPIN, resultando en:
    Costo_efectivo_FP = dias * cost_fp_per_day * (1 - compin_reversion_rate) - cost_manual_review
    """
    
    # Evaluar diferentes combinaciones de umbrales
    # Evaluar diferentes combinaciones de umbrales para optimización
    verde_candidates = np.arange(0.50, 0.999, 0.001)
    amarillo_candidates = np.arange(0.10, 0.40, 0.001)
    
    best_cost = float('inf')
    best_verde = 0.94
    best_amarillo = 0.16
    cost_details = []
    
    print("\nOptimizando umbrales basados en costos...")
    print(f"Estructura de costos:")
    print(f"  - Falso Positivo: ${cost_fp_per_day:,} por día")
    if compin_reversion_rate > 0:
        print(f"    * Ajustado por COMPIN ({compin_reversion_rate*100:.0f}% reversión)")
        print(f"    * Costo efectivo: ~${cost_fp_per_day * (1-compin_reversion_rate):,.0f} por día")
    print(f"  - Falso Negativo: ${cost_fn:,}")
    print(f"  - Revisión Manual: ${cost_manual_review:,} por caso")
    
    for verde_threshold in verde_candidates:
        for amarillo_threshold in amarillo_candidates:
            if amarillo_threshold >= verde_threshold:
                continue
                
            # Clasificar según umbrales
            verde_mask = probabilities >= verde_threshold
            amarillo_mask = (probabilities >= amarillo_threshold) & (probabilities < verde_threshold)
            rojo_mask = probabilities < amarillo_threshold
            
            # Calcular costos para zona verde (aprobación automática)
            verde_fp = ((verde_mask) & (true_labels == 0)).sum()
            verde_fn = 0  # No hay FN en verde porque aprobamos todos
            
            # Costo de FP en verde: días * costo por día
            if compin_reversion_rate > 0 and verde_fp > 0:
                # Ajustar costo considerando reversión COMPIN
                dias_fp = dias_solicitados[(verde_mask) & (true_labels == 0)]
                # Costo efectivo = dias * cost * (1 - tasa_reversion) - costo_manual_ahorrado
                verde_fp_cost = np.maximum(dias_fp * cost_fp_per_day * (1 - compin_reversion_rate) - cost_manual_review, 0).sum()
            else:
                verde_fp_cost = (dias_solicitados[(verde_mask) & (true_labels == 0)] * cost_fp_per_day).sum()
            
            # Calcular costos para zona roja (rechazo automático)
            rojo_fp = 0  # No hay FP en rojo porque rechazamos todos
            rojo_fn = ((rojo_mask) & (true_labels == 1)).sum()
            rojo_fn_cost = rojo_fn * cost_fn
            
            # CORRECCIÓN: Costo de revisión manual en zona amarilla
            # Asumimos que la revisión manual tiene una tasa de error muy baja
            # Por simplicidad, usamos solo el costo de revisión
            amarillo_cost = amarillo_mask.sum() * cost_manual_review
            
            # Costo total
            total_cost = verde_fp_cost + rojo_fn_cost + amarillo_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_verde = verde_threshold
                best_amarillo = amarillo_threshold
                
                best_details = {
                    'verde_threshold': verde_threshold,
                    'amarillo_threshold': amarillo_threshold,
                    'total_cost': total_cost,
                    'verde_cases': verde_mask.sum(),
                    'amarillo_cases': amarillo_mask.sum(),
                    'rojo_cases': rojo_mask.sum(),
                    'verde_fp': verde_fp,
                    'verde_fp_cost': verde_fp_cost,
                    'rojo_fn': rojo_fn,
                    'rojo_fn_cost': rojo_fn_cost,
                    'amarillo_cost': amarillo_cost
                }
    
    # Calcular métricas con umbrales óptimos
    verde_mask = probabilities >= best_verde
    amarillo_mask = (probabilities >= best_amarillo) & (probabilities < best_verde)
    rojo_mask = probabilities < best_amarillo
    
    # Precisión en cada zona
    verde_precision = (true_labels[verde_mask] == 1).mean() if verde_mask.sum() > 0 else 0
    rojo_precision = (true_labels[rojo_mask] == 0).mean() if rojo_mask.sum() > 0 else 0
    
    print(f"\nUmbrales óptimos encontrados:")
    print(f"  - VERDE (aprobación automática): >= {best_verde:.2f}")
    print(f"  - AMARILLO (revisión manual): {best_amarillo:.2f} - {best_verde:.2f}")
    print(f"  - ROJO (rechazo automático): < {best_amarillo:.2f}")
    
    print(f"\nDistribución de casos:")
    print(f"  - Verde: {best_details['verde_cases']:,} ({best_details['verde_cases']/len(probabilities)*100:.1f}%)")
    print(f"  - Amarillo: {best_details['amarillo_cases']:,} ({best_details['amarillo_cases']/len(probabilities)*100:.1f}%)")
    print(f"  - Rojo: {best_details['rojo_cases']:,} ({best_details['rojo_cases']/len(probabilities)*100:.1f}%)")
    
    print(f"\nCostos estimados:")
    print(f"  - Costo FP en verde: ${best_details['verde_fp_cost']:,.0f} ({best_details['verde_fp']} casos)")
    print(f"  - Costo FN en rojo: ${best_details['rojo_fn_cost']:,.0f} ({best_details['rojo_fn']} casos)")
    print(f"  - Costo revisión manual (amarillo): ${best_details['amarillo_cost']:,.0f}")
    print(f"  - COSTO TOTAL: ${best_details['total_cost']:,.0f}")
    
    print(f"\nPrecisión por zona:")
    print(f"  - Verde: {verde_precision:.1%}")
    print(f"  - Rojo: {rojo_precision:.1%}")
    
    return best_verde, best_amarillo, best_details


def apply_model_to_all_licenses(date_from=None, date_to=None,
                               optimize_thresholds=True, cost_manual_review=500,
                               compin_reversion_rate=0.0, loader=None):
    """
    Aplicar modelo a todas las licencias del período usando la estructura existente.
    
    Parámetros:
    - date_from: fecha inicial del período (None = día anterior)
    - date_to: fecha final del período (None = día anterior)
    - optimize_thresholds: si optimizar umbrales basado en costos
    - cost_manual_review: costo de revisión manual por caso
    - compin_reversion_rate: tasa de reversión del COMPIN (0.0 a 1.0)
                            Si es 0.5, significa que 50% de rechazos son revertidos
    """
    
    # Establecer fechas por defecto si no se proporcionan
    from datetime import datetime, timedelta
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if date_from is None:
        date_from = yesterday
    if date_to is None:
        date_to = yesterday
    
    print("="*80)
    print("APLICACIÓN DIARIA DE MODELO FASTTRACK 3.0")
    print("="*80)
    print("Fuente: MODELO_LM_202507_TRAIN")
    if date_from == date_to and date_from == yesterday:
        print("Procesando: SOLO licencias sin dictamen (pendientes)")
    else:
        print(f"Período: {date_from} al {date_to}")
    print("Excluyendo: PARTO y PUERPERIO")
    
    # Inicializar componentes
    created_loader = False
    if loader is None:
        loader = SnowflakeDataLoader()
        created_loader = True
    trainer = LightGBMTrainer()
    trainer.load_model('models/fasttrack_model.pkl')
    engineer = FeatureEngineer()
    engineer.load_transformers('models/feature_fasttrack.pkl')
    
    disconnect_after = False

    try:
        if loader.conn is None:
            loader.connect()
            disconnect_after = True
        else:
            disconnect_after = False
        
        # 1. CARGAR TODAS LAS LICENCIAS DEL PERÍODO
        print("\n1. Cargando licencias del período...")
        
        # SIEMPRE usar MODELO_LM_202507_TRAIN
        # Si no se especifican fechas, cargar solo licencias sin dictamen
        if date_from == date_to and date_from == yesterday:
            print("   Modo: Procesando SOLO licencias sin dictamen (pendientes)")
            query_todas = """
            SELECT *
            FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
            WHERE TARGET_APRUEBA IS NULL
            AND CIE_GRUPO NOT IN ('PARTO', 'PUERPERIO')
            AND CIE_GRUPO IS NOT NULL
            AND TRIM(CIE_GRUPO) != ''
            """
        else:
            # Si se especifican fechas, usar el filtro de fechas
            print("   Modo: Procesando licencias del período especificado")
            query_todas = f"""
            SELECT *
            FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_TRAIN
            WHERE FECHA_RECEPCION BETWEEN '{date_from}' AND '{date_to}'
            AND CIE_GRUPO NOT IN ('PARTO', 'PUERPERIO')
            AND CIE_GRUPO IS NOT NULL
            AND TRIM(CIE_GRUPO) != ''
            """
        
        df_todas = loader.execute_query(query_todas)
        print(f"   Total licencias cargadas: {len(df_todas):,}")
        
        # Separar en con dictamen y pendientes basado en TARGET_APRUEBA
        # Las pendientes son las que tienen TARGET_APRUEBA = NULL (sin dictamen aún)
        df_con_dictamen = df_todas[df_todas['TARGET_APRUEBA'].notna()].copy()
        df_pendientes_real = df_todas[df_todas['TARGET_APRUEBA'].isna()].copy()
        
        print(f"   - Con dictamen: {len(df_con_dictamen):,}")
        print(f"   - Pendientes (sin dictamen): {len(df_pendientes_real):,}")
        
        # Información adicional sobre FT1 y TARGET_FT3
        if 'LEAK_FT' in df_con_dictamen.columns:
            ft1_count = (df_con_dictamen['LEAK_FT'] == 1).sum()
            print(f"   - Pasaron por FastTrack 1.0: {ft1_count:,}")
        
        if 'TARGET_FT3' in df_con_dictamen.columns:
            target_1 = (df_con_dictamen['TARGET_FT3'] == 1).sum()
            target_0 = (df_con_dictamen['TARGET_FT3'] == 0).sum()
            target_null = df_con_dictamen['TARGET_FT3'].isna().sum()
            print(f"   - TARGET_FT3=1 (auto-aprobable): {target_1:,}")
            print(f"   - TARGET_FT3=0 (requiere revisión): {target_0:,}")
            print(f"   - TARGET_FT3=NULL (excluidas de entrenamiento): {target_null:,}")
        
        # Filtrar PARTO/PUERPERIO
        if 'CIE_GRUPO' in df_con_dictamen.columns:
            df_con_dictamen = df_con_dictamen[
                ~df_con_dictamen['CIE_GRUPO'].isin(['PARTO', 'PUERPERIO'])
            ]
        
        print(f"✓ Licencias con dictamen cargadas: {len(df_con_dictamen):,}")
        
        # Guardar información importante - Incluir TODAS las columnas necesarias para Snowflake
        columnas_info = [
            # Identificación
            'AFILIADO_RUT', 'LCC_COMCOR', 'LCC_COMCOD',
            # Columnas adicionales de identificación 
            'PRESTADOR_RUT', 'EMPLEADOR_RUT', 'N_LICENCIA',
            # Fechas
            'FECHA_RECEPCION', 'FECHA_INICIO', 'FECHA_TERMINO', 'EPISODIO_FEC_INI',
            # Información del episodio
            'EPISODIO_ACUM_DIAS', 'CONTINUA_CALC',
            # Tipo de licencia
            'TIPO_F_LM_COD', 'TIPO_F_LM',
            # Información médica
            'CIE_GRUPO', 'CIE_F', 'CIE_F_COD', 'DIASSOLICITADO',
            'LM_DIAGNOSTICO', 'LM_ANTECEDENTES_CLINICOS',
            # Demografía
            'COT_EDAD', 'COT_GENERO', 'RENTA_ESTIMADA',
            # Targets para validación
            'TARGET_FT3', 'TARGET_APRUEBA',
            # Columnas LEAK para validación
            'LEAK_FT', 'LEAK_DIASAUTORIZADOS', 'LEAK_CAUSALES', 
            'LEAK_GLOSAS', 'LEAK_ESTADOLM', 'LEAK_FALLO_PE'
        ]
        columnas_disponibles = [col for col in columnas_info if col in df_con_dictamen.columns]
        info_con_dictamen = df_con_dictamen[columnas_disponibles].copy()
        
        # 2. PREPARAR LICENCIAS PENDIENTES
        print("\n2. Preparando licencias pendientes...")
        
        # Ya tenemos df_pendientes_real del paso anterior
        df_pendientes = df_pendientes_real.copy()
        
        # IMPORTANTE: Asegurar que pendientes tiene TODAS las columnas de entrenamiento
        # Usar la función de fix_pending_features para garantizar compatibilidad
        print("   - Asegurando que df_pendientes tiene todas las columnas requeridas...")
        # Ya no es necesario - el fix se aplica después del transform
        # df_pendientes = create_complete_dataframe(df_pendientes, df_con_dictamen)
        
        # CRITICAL FIX: Since FECHA_EMP_ENVIO was NULL during training,
        # we need to set it to NaT/None for pending licenses too to ensure consistency
        if 'FECHA_EMP_ENVIO' in df_pendientes.columns and 'FECHA_EMP_ENVIO' in df_con_dictamen.columns:
            # Convert to datetime to check properly
            train_fecha = pd.to_datetime(df_con_dictamen['FECHA_EMP_ENVIO'], errors='coerce')
            if train_fecha.isna().all():
                print("   - FECHA_EMP_ENVIO was null in training, setting to None in pending for consistency")
                df_pendientes['FECHA_EMP_ENVIO'] = None
        
        print(f"✓ Licencias pendientes cargadas: {len(df_pendientes):,}")
        
        # Asegurar que las columnas de validación existan en pendientes (pueden ser NULL)
        columnas_validacion = ['TARGET_APRUEBA', 'LEAK_DIASAUTORIZADOS', 'LEAK_CAUSALES', 
                               'LEAK_GLOSAS', 'LEAK_FT', 'LEAK_ESTADOLM', 'LEAK_FALLO_PE']
        for col in columnas_validacion:
            if col not in df_pendientes.columns:
                df_pendientes[col] = None  # Añadir como NULL para pendientes
        
        # Mantener TARGET_FT3 como desconocido para pendientes (consistente con FT30)
        df_pendientes['TARGET_FT3'] = np.nan

        # Guardar información de pendientes
        columnas_disponibles_pendientes = [col for col in columnas_info if col in df_pendientes.columns]
        info_pendientes = df_pendientes[columnas_disponibles_pendientes].copy()
        
        # 3. APLICAR MODELO
        print("\n3. Aplicando modelo FastTrack 2.0...")
        
        # Crear una copia para predicción sin modificar los originales
        df_con_dictamen_pred = df_con_dictamen.copy()
        df_pendientes_pred = df_pendientes.copy()
        
        # Apply the FECHA_EMP_ENVIO fix to the prediction copy as well
        if 'FECHA_EMP_ENVIO' in df_pendientes_pred.columns and 'FECHA_EMP_ENVIO' in df_con_dictamen_pred.columns:
            train_fecha = pd.to_datetime(df_con_dictamen_pred['FECHA_EMP_ENVIO'], errors='coerce')
            pend_fecha = pd.to_datetime(df_pendientes_pred['FECHA_EMP_ENVIO'], errors='coerce')
            print(f"   - FECHA_EMP_ENVIO check: training has {train_fecha.notna().sum()} non-null, pending has {pend_fecha.notna().sum()} non-null")
            if train_fecha.isna().all() and pend_fecha.notna().any():
                print("   - Setting FECHA_EMP_ENVIO to None in pending for consistency with training")
                df_pendientes_pred['FECHA_EMP_ENVIO'] = None
                # Verify the change
                print(f"   - After fix: {df_pendientes_pred['FECHA_EMP_ENVIO'].notna().sum()} non-null values")
        
        # NO usar filtro de features - usar todas las features
        print("   - Usando TODAS las features (sin filtro IV)")
        
        # Predicciones para licencias con dictamen
        if len(df_con_dictamen_pred) > 0:
            print("   - Prediciendo licencias con dictamen...")
            try:
                X_con_dictamen = engineer.transform(df_con_dictamen_pred)
                X_con_dictamen = align_features_with_model(X_con_dictamen)
                prob_con_dictamen = trainer.predict(X_con_dictamen)  # Ya es P(TARGET_FT3=1)
                print(f"   ✓ Predicciones completadas para {len(prob_con_dictamen):,} licencias con dictamen")
            except Exception as e:
                print(f"   ✗ Error al predecir licencias con dictamen: {e}")
                raise
        else:
            print("   - No hay licencias con dictamen en este período")
            prob_con_dictamen = np.array([])
            info_con_dictamen = pd.DataFrame()  # DataFrame vacío
        
        # Predicciones para licencias pendientes
        if len(df_pendientes_pred) > 0:
            print("   - Prediciendo licencias pendientes...")
            try:
                # El feature engineering manejará automáticamente las columnas faltantes
                X_pendientes = engineer.transform(df_pendientes_pred)
                X_pendientes = align_features_with_model(X_pendientes)
                prob_pendientes = trainer.predict(X_pendientes)  # Ya es P(TARGET_FT3=1)
                print(f"   ✓ Predicciones completadas para {len(prob_pendientes):,} licencias pendientes")
            except Exception as e:
                print(f"   ✗ Error al predecir licencias pendientes: {e}")
                print("   Intentando diagnóstico detallado...")
                
                # Diagnóstico del error
                if 'feature names' in str(e).lower():
                    print("\n   Diagnóstico: Error de coincidencia de features")
                    print("   Verificando columnas en df_pendientes_pred:")
                    print(f"   - Total columnas: {len(df_pendientes_pred.columns)}")
                
                # Verificar columnas de fecha específicas
                fecha_cols = [c for c in df_pendientes_pred.columns if 'FECHA' in c]
                print(f"   - Columnas de fecha encontradas: {fecha_cols}")
                
                # Verificar si FECHA_EMP_ENVIO está presente y tiene valores
                if 'FECHA_EMP_ENVIO' in df_pendientes_pred.columns:
                    print(f"   - FECHA_EMP_ENVIO presente: {df_pendientes_pred['FECHA_EMP_ENVIO'].notna().sum()} valores no nulos")
                else:
                    print("   - FECHA_EMP_ENVIO NO está presente")
                
                raise
        else:
            print("   - No hay licencias pendientes en este período")
            prob_pendientes = np.array([])
            info_pendientes = pd.DataFrame()  # DataFrame vacío
        
        # 4. OPTIMIZAR UMBRALES (si está habilitado)
        if optimize_thresholds and 'TARGET_FT3' in info_con_dictamen.columns:
            print("\n4. Optimizando umbrales basados en costos...")
            
            # Filtrar solo casos no-FT1 para optimización
            mask_no_ft1 = info_con_dictamen['LEAK_FT'] == 0
            
            if mask_no_ft1.sum() > 0:
                optimal_verde, optimal_amarillo, cost_analysis = optimize_thresholds_by_cost(
                    probabilities=prob_con_dictamen[mask_no_ft1],
                    true_labels=info_con_dictamen.loc[mask_no_ft1, 'TARGET_FT3'],
                    dias_solicitados=info_con_dictamen.loc[mask_no_ft1, 'DIASSOLICITADO'],
                    cost_fp_per_day=59000,
                    cost_fn=20000,
                    cost_manual_review=cost_manual_review,
                    compin_reversion_rate=compin_reversion_rate
                )
                
                # Comparar con umbrales fijos
                print("\n" + "="*60)
                print("COMPARACIÓN: UMBRALES OPTIMIZADOS vs FIJOS")
                print("="*60)
                
                # Calcular costos con umbrales fijos estándar
                fixed_verde = 0.94
                fixed_amarillo = 0.16
                
                verde_mask_fixed = prob_con_dictamen[mask_no_ft1] >= fixed_verde
                amarillo_mask_fixed = (prob_con_dictamen[mask_no_ft1] >= fixed_amarillo) & (prob_con_dictamen[mask_no_ft1] < fixed_verde)
                rojo_mask_fixed = prob_con_dictamen[mask_no_ft1] < fixed_amarillo
                
                # Costos con umbrales fijos
                verde_fp_fixed = ((verde_mask_fixed) & (info_con_dictamen.loc[mask_no_ft1, 'TARGET_FT3'] == 0)).sum()
                if compin_reversion_rate > 0 and verde_fp_fixed > 0:
                    dias_fp_fixed = info_con_dictamen.loc[mask_no_ft1, 'DIASSOLICITADO'][(verde_mask_fixed) & (info_con_dictamen.loc[mask_no_ft1, 'TARGET_FT3'] == 0)]
                    verde_fp_cost_fixed = np.maximum(dias_fp_fixed * 59000 * (1 - compin_reversion_rate) - cost_manual_review, 0).sum()
                else:
                    verde_fp_cost_fixed = (info_con_dictamen.loc[mask_no_ft1, 'DIASSOLICITADO'][(verde_mask_fixed) & (info_con_dictamen.loc[mask_no_ft1, 'TARGET_FT3'] == 0)] * 59000).sum()
                rojo_fn_fixed = ((rojo_mask_fixed) & (info_con_dictamen.loc[mask_no_ft1, 'TARGET_FT3'] == 1)).sum()
                rojo_fn_cost_fixed = rojo_fn_fixed * 20000
                amarillo_cost_fixed = amarillo_mask_fixed.sum() * cost_manual_review
                total_cost_fixed = verde_fp_cost_fixed + rojo_fn_cost_fixed + amarillo_cost_fixed
                
                print(f"\nCostos con umbrales FIJOS (0.94/0.16):")
                print(f"  - Costo total: ${total_cost_fixed:,.0f}")
                print(f"\nCostos con umbrales OPTIMIZADOS ({optimal_verde:.2f}/{optimal_amarillo:.2f}):")
                print(f"  - Costo total: ${cost_analysis['total_cost']:,.0f}")
                print(f"\nAHORRO ESTIMADO: ${total_cost_fixed - cost_analysis['total_cost']:,.0f} ({(total_cost_fixed - cost_analysis['total_cost'])/total_cost_fixed*100:.1f}%)")
                
            else:
                # Umbrales estándar cuando no hay suficientes casos
                optimal_verde = 0.94
                optimal_amarillo = 0.16
                print("\nNo hay suficientes casos no-FT1 para optimizar. Usando umbrales estándar.")
        else:
            # Umbrales estándar cuando no hay optimización
            optimal_verde = 0.94
            optimal_amarillo = 0.16
            print("\n4. Usando umbrales estándar (optimización deshabilitada o sin datos de validación)")
        
        # 5. ASIGNAR SEMÁFORO
        print("\n5. Asignando clasificación por semáforo...")
        
        def asignar_semaforo(df_info, probs, verde_threshold=optimal_verde, amarillo_threshold=optimal_amarillo, context_label=""):
            """Asignar semáforo basado en probabilidades y umbrales optimizados"""
            semaforo = pd.Series('ROJO', index=df_info.index)
            
            # Procesar FT1 si existe la columna
            if 'LEAK_FT' in df_info.columns:
                # FT1: Ya aprobados
                mask_ft1 = df_info['LEAK_FT'] == 1
                semaforo[mask_ft1] = 'FT1_APROBADO'
                
                # Para los no FT1 o NULL
                mask_no_ft1 = (df_info['LEAK_FT'] != 1) | df_info['LEAK_FT'].isna()
            else:
                # Si no existe LEAK_FT, todas son no-FT1
                mask_no_ft1 = pd.Series(True, index=df_info.index)
            
            # Aplicar semáforo basado en probabilidades para todas las no-FT1
            # VERDE: Alta probabilidad de aprobación
            mask_verde = mask_no_ft1 & (probs >= verde_threshold)
            # AMARILLO: Probabilidad media
            mask_amarillo = mask_no_ft1 & (probs >= amarillo_threshold) & (probs < verde_threshold)
            # ROJO: Baja probabilidad (ya está por defecto)
            
            semaforo[mask_verde] = 'VERDE'
            semaforo[mask_amarillo] = 'AMARILLO'

            if 'DIASSOLICITADO' in df_info.columns:
                dias_solicitados = pd.to_numeric(df_info['DIASSOLICITADO'], errors='coerce')
            else:
                dias_solicitados = pd.Series(np.nan, index=df_info.index)

            if 'EPISODIO_ACUM_DIAS' in df_info.columns:
                episodio_acum = pd.to_numeric(df_info['EPISODIO_ACUM_DIAS'], errors='coerce')
            else:
                episodio_acum = pd.Series(np.nan, index=df_info.index)

            regla_negocio_mask = mask_verde & (
                (dias_solicitados > 20) | (episodio_acum >= 30)
            )

            if regla_negocio_mask.any():
                semaforo[regla_negocio_mask] = 'AMARILLO'
                contexto_msg = f" ({context_label})" if context_label else ""
                print(
                    f"      ⚠ Regla de negocio aplicada{contexto_msg}: {regla_negocio_mask.sum():,} licencias con más de 20 días solicitados o >=30 días acumulados forzadas a AMARILLO"
                )

            return semaforo
        
        # Aplicar clasificación con umbrales optimizados
        if len(info_con_dictamen) > 0:
            info_con_dictamen['PROBABILIDAD_APROBACION'] = prob_con_dictamen
            info_con_dictamen['SEMAFORO'] = asignar_semaforo(
                info_con_dictamen,
                prob_con_dictamen,
                optimal_verde,
                optimal_amarillo,
                context_label='licencias con dictamen'
            )
            info_con_dictamen['UMBRAL_VERDE'] = optimal_verde
            info_con_dictamen['UMBRAL_AMARILLO'] = optimal_amarillo
        
        if len(info_pendientes) > 0:
            info_pendientes['PROBABILIDAD_APROBACION'] = prob_pendientes
            info_pendientes['SEMAFORO'] = asignar_semaforo(
                info_pendientes,
                prob_pendientes,
                optimal_verde,
                optimal_amarillo,
                context_label='licencias pendientes'
            )
            info_pendientes['UMBRAL_VERDE'] = optimal_verde
            info_pendientes['UMBRAL_AMARILLO'] = optimal_amarillo
        
        # Ordenar las pendientes para Snowflake y Excel
        if len(info_pendientes) > 0:
            info_pendientes_sorted = info_pendientes.sort_values(
                ['SEMAFORO', 'PROBABILIDAD_APROBACION'], 
                ascending=[True, False]
            )
        else:
            info_pendientes_sorted = info_pendientes  # DataFrame vacío
        
        # 6. ANÁLISIS DE RESULTADOS
        print("\n" + "="*80)
        print("RESUMEN DE RESULTADOS")
        print("="*80)
        print(f"\nUMBRALES UTILIZADOS:")
        print(f"  - Verde (aprobación): >= {optimal_verde:.2f}")
        print(f"  - Amarillo (revisión): {optimal_amarillo:.2f} - {optimal_verde:.2f}")
        print(f"  - Rojo (rechazo): < {optimal_amarillo:.2f}")
        
        total_licencias = len(info_con_dictamen) + len(info_pendientes)
        print(f"\nTOTAL LICENCIAS PROCESADAS: {total_licencias:,}")
        print(f"  - Con dictamen: {len(info_con_dictamen):,}")
        print(f"  - Pendientes: {len(info_pendientes):,}")
        
        # Validación con licencias con dictamen
        print("\n\nA. VALIDACIÓN DEL MODELO (licencias con dictamen):")
        print("-" * 50)
        
        if len(info_con_dictamen) > 0:
            for sem in ['FT1_APROBADO', 'VERDE', 'AMARILLO', 'ROJO']:
                n = (info_con_dictamen['SEMAFORO'] == sem).sum()
                if n > 0:
                    pct = n / len(info_con_dictamen) * 100
                    print(f"  {sem}: {n:,} ({pct:.1f}%)")
        else:
            print("  No hay licencias con dictamen en este período")
        
        # Métricas zona verde
        if len(info_con_dictamen) > 0 and 'SEMAFORO' in info_con_dictamen.columns:
            verde_con_dictamen = info_con_dictamen[info_con_dictamen['SEMAFORO'] == 'VERDE']
        else:
            verde_con_dictamen = pd.DataFrame()
        
        if len(verde_con_dictamen) > 0:
            tp = (verde_con_dictamen['TARGET_FT3'] == 1).sum()
            fp = (verde_con_dictamen['TARGET_FT3'] == 0).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"\nMétricas Zona VERDE:")
            print(f"  Total: {len(verde_con_dictamen):,}")
            print(f"  Precisión: {precision:.1%}")
            print(f"  - Verdaderos positivos: {tp:,}")
            print(f"  - Falsos positivos: {fp:,}")
            
        # Predicciones para pendientes
        print("\n\nB. PREDICCIONES PARA LICENCIAS PENDIENTES:")
        print("-" * 50)
        print(f"Total pendientes: {len(info_pendientes):,}")
        
        if len(info_pendientes) > 0 and 'SEMAFORO' in info_pendientes.columns:
            for sem in ['FT1_APROBADO', 'VERDE', 'AMARILLO', 'ROJO']:
                n = (info_pendientes['SEMAFORO'] == sem).sum()
                if n > 0:
                    pct = n / len(info_pendientes) * 100
                    print(f"  {sem}: {n:,} ({pct:.1f}%)")
        
        # Detalle zona verde pendientes
        if len(info_pendientes) > 0 and 'SEMAFORO' in info_pendientes.columns:
            verde_pendientes = info_pendientes[info_pendientes['SEMAFORO'] == 'VERDE']
        else:
            verde_pendientes = pd.DataFrame()
            
        if len(verde_pendientes) > 0:
            print(f"\nRecomendadas para aprobación (VERDE): {len(verde_pendientes):,}")
            print(f"Días solicitados totales: {verde_pendientes['DIASSOLICITADO'].sum():,}")
            
            print("\nTop 10 diagnósticos en zona VERDE:")
            for i, (cie, count) in enumerate(verde_pendientes['CIE_GRUPO'].value_counts().head(10).items(), 1):
                pct = count / len(verde_pendientes) * 100
                print(f"  {i:2d}. {cie}: {count:,} ({pct:.1f}%)")
        
        # 7. GUARDAR RESULTADOS EN DATABRICKS SQL
        print("\n\n7. Guardando predicciones en Databricks SQL...")
        
        # Generar timestamp para el archivo Excel
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fecha_procesamiento = datetime.now()
        
        # Guardar predicciones diarias (todas: con dictamen y pendientes)
        try:
            # 7.1 GUARDAR PREDICCIONES DIARIAS (TODAS)
            print("   A. Guardando predicciones diarias (todas las licencias)...")
            
            # Combinar todas las predicciones
            all_predictions = pd.concat([info_con_dictamen, info_pendientes], ignore_index=True)
            
            # Debug: Ver qué columnas tiene all_predictions
            print(f"      Columnas en all_predictions: {list(all_predictions.columns)}")
            print(f"      Total columnas: {len(all_predictions.columns)}")
            
            # Verificar columnas específicas problemáticas
            columnas_problema = ['PRESTADOR_RUT', 'EMPLEADOR_RUT', 'N_LICENCIA', 'TIPO_F_LM_COD', 
                                'FECHA_INICIO', 'FECHA_TERMINO', 'CIE_F_COD']
            for col in columnas_problema:
                if col in all_predictions.columns:
                    print(f"      ✓ {col} existe con {all_predictions[col].notna().sum()} valores no nulos")
                else:
                    print(f"      ✗ {col} NO existe")
            
            # Preparar para Databricks SQL
            df_daily = all_predictions.copy()
            df_daily['FECHA_PROCESAMIENTO'] = fecha_procesamiento
            df_daily['FECHA_DESDE'] = date_from
            df_daily['FECHA_HASTA'] = date_to
            df_daily['MODELO_VERSION'] = 'FT30'
            df_daily['ES_PENDIENTE'] = df_daily['LCC_COMCOR'].isin(info_pendientes['LCC_COMCOR']).astype(int)
            
            # Convertir campos de fecha a formato string ISO para evitar problemas de carga
            date_columns = ['FECHA_RECEPCION', 'FECHA_INICIO', 'FECHA_TERMINO', 'EPISODIO_FEC_INI']
            for col in date_columns:
                if col in df_daily.columns:
                    # Convertir a datetime primero
                    df_daily[col] = pd.to_datetime(df_daily[col], errors='coerce')
                    # Convertir a string ISO format (YYYY-MM-DD HH:MM:SS) o None para NaT
                    df_daily[col] = df_daily[col].dt.strftime('%Y-%m-%d %H:%M:%S').where(pd.notnull(df_daily[col]), None)
            
            # Tabla de predicciones diarias
            tabla_diaria = "FT30_PREDICCIONES_DIARIAS"
            
            # Verificar si la tabla existe
            cursor = loader.conn.cursor()
            
            # Crear tabla si no existe
            create_daily_table = f"""
            CREATE TABLE IF NOT EXISTS {tabla_diaria} (
                -- Primeros 20 campos (sin LCC_OPERADOR que no existe)
                AFILIADO_RUT VARCHAR(11),
                LCC_COMCOD NUMBER(38,0),
                LCC_COMCOR NUMBER(38,0),
                LCC_OPERADOR NUMBER(38,0),  -- NULL siempre, no existe en datos
                LCC_MEDRUT VARCHAR(11),     -- Viene de PRESTADOR_RUT
                LCC_EMPRUT VARCHAR(11),     -- Viene de EMPLEADOR_RUT
                LCC_IDN VARCHAR(20),        -- Ampliado para acomodar valores más largos
                EPISODIO_ACUM_DIAS NUMBER(38,0),
                EPISODIO_FEC_INI DATE,
                CONTINUA_CALC VARCHAR(1),
                TIPO_F_LM_COD NUMBER(38,0),
                TIPO_F_LM VARCHAR(35),
                FECHA_RECEPCION TIMESTAMP_NTZ(3),
                FECHA_INICIO TIMESTAMP_NTZ(3),
                FECHA_TERMINO TIMESTAMP_NTZ(3),
                DIASSOLICITADO NUMBER(38,0),
                CIE_F VARCHAR(120),
                CIE_F_COD VARCHAR(12),
                CIE_GRUPO VARCHAR(30),
                LM_DIAGNOSTICO VARCHAR(100),
                LM_ANTECEDENTES_CLINICOS VARCHAR(100),
                -- Campos adicionales de afiliado
                COT_EDAD NUMBER(38,0),
                COT_GENERO VARCHAR(4),
                RENTA_ESTIMADA FLOAT,
                -- Campos del modelo
                SEMAFORO VARCHAR(20),
                PROBABILIDAD_APROBACION FLOAT,
                UMBRAL_VERDE FLOAT,
                UMBRAL_AMARILLO FLOAT,
                FECHA_PROCESAMIENTO TIMESTAMP_NTZ,
                FECHA_DESDE DATE,
                FECHA_HASTA DATE,
                MODELO_VERSION VARCHAR(10),
                ES_PENDIENTE NUMBER(1,0),
                TARGET_FT3 NUMBER(1,0),
                TARGET_APRUEBA NUMBER(1,0),
                GLOSA_GENERADA VARCHAR(500),
                CAUSAL_GENERADA VARCHAR(100),
                -- Campos LEAK para validación
                LEAK_FT VARCHAR(100),
                LEAK_CAUSALES VARCHAR(500),
                LEAK_DIASAUTORIZADOS NUMBER(38,0),
                LEAK_GLOSAS VARCHAR(500),
                LEAK_ESTADOLM VARCHAR(35),
                LEAK_FALLO_PE VARCHAR(255)
            )
            """
            
            cursor.execute(create_daily_table)
            print(f"      ✓ Tabla {tabla_diaria} creada/verificada")
            
            # Crear tabla temporal para el MERGE
            tabla_temp = f"{tabla_diaria}_TEMP"
            
            # Eliminar tabla temporal si existe
            cursor.execute(f"DROP TABLE IF EXISTS {tabla_temp}")
            
            # Crear tabla temporal con la misma estructura
            cursor.execute(f"CREATE TABLE {tabla_temp} LIKE {tabla_diaria}")
            print(f"      ✓ Tabla temporal {tabla_temp} creada")
            
            # Agregar campos adicionales del modelo
            df_daily['GLOSA_GENERADA'] = ''
            df_daily['CAUSAL_GENERADA'] = ''
            
            # Generar glosas según el semáforo
            df_daily.loc[df_daily['SEMAFORO'] == 'VERDE', 'GLOSA_GENERADA'] = 'Licencia aprobada automáticamente por modelo predictivo'
            df_daily.loc[df_daily['SEMAFORO'] == 'AMARILLO', 'GLOSA_GENERADA'] = 'Licencia requiere revisión manual'
            df_daily.loc[df_daily['SEMAFORO'] == 'ROJO', 'GLOSA_GENERADA'] = 'Licencia rechazada automáticamente por modelo predictivo'
            
            # IMPORTANTE: Las columnas ya vienen de MODELO_LM_202507_TRAIN
            # Solo necesitamos mapear las que tienen nombres diferentes
            
            # === PASO 1: HACER TODOS LOS MAPEOS Y CREAR COLUMNAS FALTANTES ===
            
            # Mapeos de columnas con nombres diferentes
            if 'PRESTADOR_RUT' in df_daily.columns:
                df_daily['LCC_MEDRUT'] = df_daily['PRESTADOR_RUT']
                print(f"      ✓ Mapeado PRESTADOR_RUT -> LCC_MEDRUT: {df_daily['LCC_MEDRUT'].notna().sum()} valores")
            else:
                df_daily['LCC_MEDRUT'] = pd.Series([None] * len(df_daily), index=df_daily.index)
                print(f"      ⚠️ PRESTADOR_RUT no existe, LCC_MEDRUT creado vacío")
                
            if 'EMPLEADOR_RUT' in df_daily.columns:
                df_daily['LCC_EMPRUT'] = df_daily['EMPLEADOR_RUT']
                print(f"      ✓ Mapeado EMPLEADOR_RUT -> LCC_EMPRUT: {df_daily['LCC_EMPRUT'].notna().sum()} valores")
            else:
                df_daily['LCC_EMPRUT'] = pd.Series([None] * len(df_daily), index=df_daily.index)
                print(f"      ⚠️ EMPLEADOR_RUT no existe, LCC_EMPRUT creado vacío")
            
            # N_LICENCIA -> LCC_IDN
            if 'N_LICENCIA' in df_daily.columns:
                df_daily['LCC_IDN'] = df_daily['N_LICENCIA']
                print(f"      ✓ Mapeado N_LICENCIA -> LCC_IDN: {df_daily['LCC_IDN'].notna().sum()} valores")
            else:
                df_daily['LCC_IDN'] = pd.Series([None] * len(df_daily), index=df_daily.index)
                print(f"      ⚠️ N_LICENCIA no existe, LCC_IDN creado vacío")
                
            # LCC_OPERADOR no existe en los datos fuente - crear vacío
            df_daily['LCC_OPERADOR'] = pd.Series([None] * len(df_daily), index=df_daily.index)
            
            # COT_EDAD y COT_GENERO ya vienen de MODELO_LM_202507_TRAIN
            # RENTA_ESTIMADA también viene de la tabla
            # Solo crear si realmente no existen
            if 'COT_EDAD' not in df_daily.columns:
                print("      ⚠️ COT_EDAD no encontrado - creando con NULL")
                df_daily['COT_EDAD'] = pd.Series([None] * len(df_daily), index=df_daily.index)
                    
            if 'COT_GENERO' not in df_daily.columns:
                print("      ⚠️ COT_GENERO no encontrado - creando con NULL")
                df_daily['COT_GENERO'] = pd.Series([None] * len(df_daily), index=df_daily.index)
                    
            if 'RENTA_ESTIMADA' not in df_daily.columns:
                print("      ⚠️ RENTA_ESTIMADA no encontrado - creando con 0")
                df_daily['RENTA_ESTIMADA'] = 0
            
            # Crear columnas LEAK si no existen
            leak_columns_list = ['LEAK_FT', 'LEAK_CAUSALES', 'LEAK_DIASAUTORIZADOS', 
                                'LEAK_GLOSAS', 'LEAK_ESTADOLM', 'LEAK_FALLO_PE']
            for col in leak_columns_list:
                if col not in df_daily.columns:
                    df_daily[col] = pd.Series([None] * len(df_daily), index=df_daily.index)
                    print(f"      ✓ Creada columna {col} con valores NULL")
            
            # Debug: Verificar si las columnas críticas tienen datos
            if 'LCC_COMCOD' in df_daily.columns:
                print(f"      LCC_COMCOD: {df_daily['LCC_COMCOD'].notna().sum()} valores no nulos de {len(df_daily)}")
            else:
                print(f"      ⚠️ LCC_COMCOD NO existe en df_daily")
                
            if 'LCC_COMCOR' in df_daily.columns:
                print(f"      LCC_COMCOR: {df_daily['LCC_COMCOR'].notna().sum()} valores no nulos de {len(df_daily)}")
            else:
                print(f"      ⚠️ LCC_COMCOR NO existe en df_daily")
                
            # === PASO 2: AHORA CONSTRUIR LA LISTA DE COLUMNAS DESPUÉS DE TODOS LOS MAPEOS ===
            
            # Lista de columnas - usar nombres que YA EXISTEN en el DataFrame después de mapeos
            # Columnas base de la tabla ALFIL/MODELO_LM_202507_TRAIN
            base_columns = [
                'AFILIADO_RUT',           # Existe
                'LCC_COMCOD',            # Revisar si existe
                'LCC_COMCOR',            # Existe  
                'LCC_OPERADOR',          # YA CREADO arriba
                'LCC_MEDRUT',            # YA CREADO desde PRESTADOR_RUT
                'LCC_EMPRUT',            # YA CREADO desde EMPLEADOR_RUT
                'LCC_IDN',               # YA CREADO desde N_LICENCIA
                'EPISODIO_ACUM_DIAS',    # Revisar
                'EPISODIO_FEC_INI',      # Revisar
                'CONTINUA_CALC',         # Revisar
                'TIPO_F_LM_COD',         # Revisar
                'TIPO_F_LM',             # Revisar
                'FECHA_RECEPCION',       # Existe
                'FECHA_INICIO',          # Revisar
                'FECHA_TERMINO',         # Revisar
                'DIASSOLICITADO',        # Existe
                'CIE_F',                 # Existe
                'CIE_F_COD',             # Revisar
                'CIE_GRUPO',             # Existe
                'LM_DIAGNOSTICO',        # Revisar
                'LM_ANTECEDENTES_CLINICOS', # Revisar
                # NO incluir FECHA_EMISION_DT - no existe en la tabla destino
                'COT_EDAD',              # YA CREADO
                'COT_GENERO',            # YA CREADO
                'RENTA_ESTIMADA'         # YA CREADO
            ]
            
            # Campos del modelo y control
            model_columns = [
                'SEMAFORO', 'PROBABILIDAD_APROBACION', 'UMBRAL_VERDE', 'UMBRAL_AMARILLO',
                'FECHA_PROCESAMIENTO', 'FECHA_DESDE', 'FECHA_HASTA', 'MODELO_VERSION', 
                'ES_PENDIENTE', 'TARGET_FT3', 'TARGET_APRUEBA',
                'GLOSA_GENERADA', 'CAUSAL_GENERADA'
            ]
            
            # Campos LEAK (validación) - YA CREADOS arriba
            leak_columns = [
                'LEAK_FT', 'LEAK_CAUSALES', 'LEAK_DIASAUTORIZADOS', 
                'LEAK_GLOSAS', 'LEAK_ESTADOLM', 'LEAK_FALLO_PE'
            ]
            
            # IMPORTANTE: Ahora incluir columnas, creando las que falten
            columns_to_save = []
            all_columns = base_columns + model_columns + leak_columns
            
            print(f"      Verificando {len(all_columns)} columnas para guardar...")
            for col in all_columns:
                if col in df_daily.columns:
                    columns_to_save.append(col)
                    valores = df_daily[col].notna().sum()
                    if valores == 0:
                        print(f"      ⚠️ {col} existe pero está vacía (0 valores)")
                else:
                    # Si no existe, crearla vacía y agregarla a la lista
                    print(f"      ✗ {col} NO existe en df_daily - creando vacía")
                    df_daily[col] = pd.Series([None] * len(df_daily), index=df_daily.index)
                    columns_to_save.append(col)
            
            # Truncar campos de texto para evitar exceder límites de columnas
            if 'CIE_GRUPO' in df_daily.columns:
                df_daily['CIE_GRUPO'] = df_daily['CIE_GRUPO'].astype(str).str[:30]
            if 'CIE_F' in df_daily.columns:
                df_daily['CIE_F'] = df_daily['CIE_F'].astype(str).str[:120]
            if 'GLOSA_GENERADA' in df_daily.columns:
                df_daily['GLOSA_GENERADA'] = df_daily['GLOSA_GENERADA'].astype(str).str[:500]
            if 'CAUSAL_GENERADA' in df_daily.columns:
                df_daily['CAUSAL_GENERADA'] = df_daily['CAUSAL_GENERADA'].astype(str).str[:100]
            
            # Convertir columnas de fecha a formato string ISO para carga
            date_columns = ['FECHA_RECEPCION', 'FECHA_INICIO', 'FECHA_TERMINO', 'EPISODIO_FEC_INI', 
                          'FECHA_PROCESAMIENTO', 'FECHA_DESDE', 'FECHA_HASTA']
            for col in date_columns:
                if col in df_daily.columns:
                    # Convertir a datetime primero
                    df_daily[col] = pd.to_datetime(df_daily[col], errors='coerce')
                    # Convertir a string ISO format (YYYY-MM-DD HH:MM:SS) o None para NaT
                    df_daily[col] = df_daily[col].dt.strftime('%Y-%m-%d %H:%M:%S').where(pd.notnull(df_daily[col]), None)
            
            # Debug: verificar qué columnas realmente tenemos antes de guardar
            print(f"      Columnas a guardar: {len(columns_to_save)}")
            print(f"      Columnas en df_daily: {len(df_daily.columns)}")
            
            # Crear un DataFrame temporal solo con las columnas que vamos a guardar
            df_to_save = df_daily[columns_to_save].copy()
            
            # Debug: verificar que el DataFrame a guardar tiene las columnas mapeadas
            print(f"      Columnas en df_to_save: {list(df_to_save.columns)[:10]}...")
            columnas_verificar = ['LCC_MEDRUT', 'LCC_EMPRUT', 'LCC_IDN', 'COT_EDAD']
            for col in columnas_verificar:
                if col in df_to_save.columns:
                    valores = df_to_save[col].notna().sum()
                    print(f"      ✓ {col} está en df_to_save con {valores} valores no nulos")
                else:
                    print(f"      ⚠️ {col} NO está en df_to_save")
            
            # Insertar datos en tabla temporal
            success, nchunks, nrows, _ = _write_dataframe_to_table(
                conn=loader.conn,
                df=df_to_save,
                table_name=tabla_temp,
                database=loader.connection_params['database'],
                schema=loader.connection_params['schema'],
                quote_identifiers=False,
                auto_create_table=False,
                overwrite=False
            )
            
            if success:
                print(f"      ✓ {nrows:,} registros cargados en tabla temporal")
                
                # Hacer MERGE de tabla temporal a tabla final
                merge_query = f"""
                MERGE INTO {tabla_diaria} AS target
                USING {tabla_temp} AS source
                ON target.AFILIADO_RUT = source.AFILIADO_RUT 
                   AND target.LCC_COMCOR = source.LCC_COMCOR
                WHEN MATCHED THEN
                    UPDATE SET
                        LCC_COMCOD = source.LCC_COMCOD,
                        LCC_OPERADOR = source.LCC_OPERADOR,
                        LCC_MEDRUT = source.LCC_MEDRUT,
                        LCC_EMPRUT = source.LCC_EMPRUT,
                        LCC_IDN = source.LCC_IDN,
                        EPISODIO_ACUM_DIAS = source.EPISODIO_ACUM_DIAS,
                        EPISODIO_FEC_INI = source.EPISODIO_FEC_INI,
                        CONTINUA_CALC = source.CONTINUA_CALC,
                        TIPO_F_LM_COD = source.TIPO_F_LM_COD,
                        TIPO_F_LM = source.TIPO_F_LM,
                        FECHA_RECEPCION = source.FECHA_RECEPCION,
                        FECHA_INICIO = source.FECHA_INICIO,
                        FECHA_TERMINO = source.FECHA_TERMINO,
                        DIASSOLICITADO = source.DIASSOLICITADO,
                        CIE_F = source.CIE_F,
                        CIE_F_COD = source.CIE_F_COD,
                        CIE_GRUPO = source.CIE_GRUPO,
                        LM_DIAGNOSTICO = source.LM_DIAGNOSTICO,
                        LM_ANTECEDENTES_CLINICOS = source.LM_ANTECEDENTES_CLINICOS,
                        COT_EDAD = source.COT_EDAD,
                        COT_GENERO = source.COT_GENERO,
                        RENTA_ESTIMADA = source.RENTA_ESTIMADA,
                        SEMAFORO = COALESCE(target.SEMAFORO, source.SEMAFORO),
                        PROBABILIDAD_APROBACION = COALESCE(target.PROBABILIDAD_APROBACION, source.PROBABILIDAD_APROBACION),
                        UMBRAL_VERDE = COALESCE(target.UMBRAL_VERDE, source.UMBRAL_VERDE),
                        UMBRAL_AMARILLO = COALESCE(target.UMBRAL_AMARILLO, source.UMBRAL_AMARILLO),
                        FECHA_PROCESAMIENTO = COALESCE(target.FECHA_PROCESAMIENTO, source.FECHA_PROCESAMIENTO),
                        FECHA_DESDE = COALESCE(target.FECHA_DESDE, source.FECHA_DESDE),
                        FECHA_HASTA = COALESCE(target.FECHA_HASTA, source.FECHA_HASTA),
                        MODELO_VERSION = COALESCE(target.MODELO_VERSION, source.MODELO_VERSION),
                        ES_PENDIENTE = source.ES_PENDIENTE,
                        TARGET_FT3 = source.TARGET_FT3,
                        TARGET_APRUEBA = source.TARGET_APRUEBA,
                        GLOSA_GENERADA = COALESCE(target.GLOSA_GENERADA, source.GLOSA_GENERADA),
                        CAUSAL_GENERADA = COALESCE(target.CAUSAL_GENERADA, source.CAUSAL_GENERADA),
                        LEAK_FT = source.LEAK_FT,
                        LEAK_CAUSALES = source.LEAK_CAUSALES,
                        LEAK_DIASAUTORIZADOS = source.LEAK_DIASAUTORIZADOS,
                        LEAK_GLOSAS = source.LEAK_GLOSAS,
                        LEAK_ESTADOLM = source.LEAK_ESTADOLM,
                        LEAK_FALLO_PE = source.LEAK_FALLO_PE
                WHEN NOT MATCHED THEN
                    INSERT (AFILIADO_RUT, LCC_COMCOD, LCC_COMCOR, LCC_OPERADOR,
                           LCC_MEDRUT, LCC_EMPRUT, LCC_IDN,
                           EPISODIO_ACUM_DIAS, EPISODIO_FEC_INI, CONTINUA_CALC,
                           TIPO_F_LM_COD, TIPO_F_LM,
                           FECHA_RECEPCION, FECHA_INICIO, FECHA_TERMINO, DIASSOLICITADO,
                           CIE_F, CIE_F_COD, CIE_GRUPO, LM_DIAGNOSTICO, LM_ANTECEDENTES_CLINICOS,
                           COT_EDAD, COT_GENERO, RENTA_ESTIMADA,
                           SEMAFORO, PROBABILIDAD_APROBACION, UMBRAL_VERDE, UMBRAL_AMARILLO,
                           FECHA_PROCESAMIENTO, FECHA_DESDE, FECHA_HASTA, MODELO_VERSION,
                           ES_PENDIENTE, TARGET_FT3, TARGET_APRUEBA,
                           GLOSA_GENERADA, CAUSAL_GENERADA,
                           LEAK_FT, LEAK_CAUSALES, LEAK_DIASAUTORIZADOS,
                           LEAK_GLOSAS, LEAK_ESTADOLM, LEAK_FALLO_PE)
                    VALUES (source.AFILIADO_RUT, source.LCC_COMCOD, source.LCC_COMCOR, source.LCC_OPERADOR,
                           source.LCC_MEDRUT, source.LCC_EMPRUT, source.LCC_IDN,
                           source.EPISODIO_ACUM_DIAS, source.EPISODIO_FEC_INI, source.CONTINUA_CALC,
                           source.TIPO_F_LM_COD, source.TIPO_F_LM,
                           source.FECHA_RECEPCION, source.FECHA_INICIO, source.FECHA_TERMINO, source.DIASSOLICITADO,
                           source.CIE_F, source.CIE_F_COD, source.CIE_GRUPO, source.LM_DIAGNOSTICO, source.LM_ANTECEDENTES_CLINICOS,
                           source.COT_EDAD, source.COT_GENERO, source.RENTA_ESTIMADA,
                           source.SEMAFORO, source.PROBABILIDAD_APROBACION, source.UMBRAL_VERDE, source.UMBRAL_AMARILLO,
                           source.FECHA_PROCESAMIENTO, source.FECHA_DESDE, source.FECHA_HASTA, source.MODELO_VERSION,
                           source.ES_PENDIENTE, source.TARGET_FT3, source.TARGET_APRUEBA,
                           source.GLOSA_GENERADA, source.CAUSAL_GENERADA,
                           source.LEAK_FT, source.LEAK_CAUSALES, source.LEAK_DIASAUTORIZADOS,
                           source.LEAK_GLOSAS, source.LEAK_ESTADOLM, source.LEAK_FALLO_PE)
                """
                
                cursor.execute(merge_query)
                merge_count = cursor.rowcount
                print(f"      ✓ MERGE completado: {merge_count} registros afectados")
                
                # Limpiar tabla temporal
                cursor.execute(f"DROP TABLE IF EXISTS {tabla_temp}")
                
                # Verificar inserción
                verify_query = f"""
                SELECT COUNT(*) as total,
                       SUM(ES_PENDIENTE) as pendientes,
                       COUNT(DISTINCT SEMAFORO) as semaforos
                FROM {tabla_diaria}
                WHERE DATE(FECHA_PROCESAMIENTO) = CURRENT_DATE()
                """
                cursor.execute(verify_query)
                result = cursor.fetchone()
                if result:
                    total_count = result[0] if result[0] is not None else 0
                    pending_count = int(result[1]) if result[1] is not None else 0
                    semaforos_count = result[2] if result[2] is not None else 0
                    print(f"      ✓ Verificación: {total_count:,} total, {pending_count:,} pendientes, {semaforos_count} semáforos")
            
            # 7.2 GUARDAR TAMBIÉN LICENCIAS PENDIENTES EN TABLA SEPARADA (compatibilidad)
            print("\n   B. Guardando licencias pendientes en tabla separada...")
            
            # Preparar datos para Databricks SQL - solo licencias pendientes
            df_snowflake_pendientes = info_pendientes_sorted.copy()
            
            # IMPORTANTE: Asegurar que LCC_COMCOR sea string ANTES de cualquier procesamiento
            df_snowflake_pendientes['LCC_COMCOR'] = df_snowflake_pendientes['LCC_COMCOR'].astype(str)
            
            # Agregar columna de fecha de procesamiento
            df_snowflake_pendientes['FECHA_PROCESAMIENTO'] = fecha_procesamiento
            df_snowflake_pendientes['FECHA_PROCESAMIENTO_STR'] = fecha_procesamiento.strftime('%Y-%m-%d %H:%M:%S')
            
            # Convertir campos de fecha a formato string ISO para carga
            date_columns = ['FECHA_RECEPCION', 'FECHA_INICIO', 'FECHA_TERMINO', 'EPISODIO_FEC_INI', 'FECHA_PROCESAMIENTO']
            for col in date_columns:
                if col in df_snowflake_pendientes.columns:
                    # Convertir a datetime primero
                    df_snowflake_pendientes[col] = pd.to_datetime(df_snowflake_pendientes[col], errors='coerce')
                    # Convertir a string ISO format (YYYY-MM-DD HH:MM:SS) o None para NaT
                    df_snowflake_pendientes[col] = df_snowflake_pendientes[col].dt.strftime('%Y-%m-%d %H:%M:%S').where(pd.notnull(df_snowflake_pendientes[col]), None)
            
            # Nombre de la tabla fija (sin timestamp)
            tabla_pendientes = "FT30_LICENCIAS_PENDIENTES"
            
            # Verificar si la tabla existe y obtener licencias ya procesadas
            cursor = loader.conn.cursor()
            
            # Verificar si la tabla existe
            cursor.execute(f"SHOW TABLES LIKE '{tabla_pendientes}'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Si la tabla no existe, crearla con el schema correcto
                print(f"   - Creando tabla {tabla_pendientes} por primera vez...")
                create_pendientes_table = f"""
                CREATE TABLE {tabla_pendientes} (
                    -- Primeros 20 campos (sin LCC_OPERADOR que no existe)
                    AFILIADO_RUT VARCHAR(11),
                    LCC_COMCOD NUMBER(38,0),
                    LCC_COMCOR NUMBER(38,0),
                    LCC_OPERADOR NUMBER(38,0),  -- NULL siempre, no existe en datos
                    LCC_MEDRUT VARCHAR(11),     -- Viene de PRESTADOR_RUT
                    LCC_EMPRUT VARCHAR(11),     -- Viene de EMPLEADOR_RUT
                    LCC_IDN VARCHAR(20),        -- Ampliado para acomodar valores más largos
                    EPISODIO_ACUM_DIAS NUMBER(38,0),
                    EPISODIO_FEC_INI DATE,
                    CONTINUA_CALC VARCHAR(1),
                    TIPO_F_LM_COD NUMBER(38,0),
                    TIPO_F_LM VARCHAR(35),
                    FECHA_RECEPCION TIMESTAMP_NTZ(3),
                    FECHA_INICIO TIMESTAMP_NTZ(3),
                    FECHA_TERMINO TIMESTAMP_NTZ(3),
                    DIASSOLICITADO NUMBER(38,0),
                    CIE_F VARCHAR(120),
                    CIE_F_COD VARCHAR(12),
                    CIE_GRUPO VARCHAR(30),
                    LM_DIAGNOSTICO VARCHAR(100),
                    LM_ANTECEDENTES_CLINICOS VARCHAR(100),
                    -- Campos adicionales de afiliado
                    COT_EDAD NUMBER(38,0),
                    COT_GENERO VARCHAR(4),
                    RENTA_ESTIMADA FLOAT,
                    -- Campos del modelo
                    SEMAFORO VARCHAR(20),
                    PROBABILIDAD_APROBACION FLOAT,
                    UMBRAL_VERDE FLOAT,
                    UMBRAL_AMARILLO FLOAT,
                    FECHA_PROCESAMIENTO TIMESTAMP_NTZ,
                    FECHA_PROCESAMIENTO_STR VARCHAR(50),
                    MODELO_VERSION VARCHAR(10),
                    TARGET_FT3 NUMBER(1,0),
                    TARGET_APRUEBA NUMBER(1,0),
                    GLOSA_GENERADA VARCHAR(500),
                    CAUSAL_GENERADA VARCHAR(100),
                    -- Campos LEAK para validación
                    LEAK_FT VARCHAR(100),
                    LEAK_CAUSALES VARCHAR(500),
                    LEAK_DIASAUTORIZADOS NUMBER(38,0),
                    LEAK_GLOSAS VARCHAR(500),
                    LEAK_ESTADOLM VARCHAR(35),
                    LEAK_FALLO_PE VARCHAR(255)
                )
                """
                cursor.execute(create_pendientes_table)
                print(f"      ✓ Tabla {tabla_pendientes} creada")
            else:
                print(f"   - Tabla {tabla_pendientes} ya existe")
            
            # Crear tabla temporal para MERGE de pendientes
            tabla_temp_pend = f"{tabla_pendientes}_TEMP"
            cursor.execute(f"DROP TABLE IF EXISTS {tabla_temp_pend}")
            cursor.execute(f"CREATE TABLE {tabla_temp_pend} LIKE {tabla_pendientes}")
            
            if len(df_snowflake_pendientes) > 0:
                print(f"   - Procesando {len(df_snowflake_pendientes):,} licencias pendientes...")
                
                # Cargar datos en tabla temporal
                success, nchunks, nrows, _ = _write_dataframe_to_table(
                    conn=loader.conn,
                    df=df_snowflake_pendientes,
                    table_name=tabla_temp_pend,
                    database=loader.connection_params['database'],
                    schema=loader.connection_params['schema'],
                    quote_identifiers=False,
                    auto_create_table=True,
                    overwrite=False
                )
                
                if success:
                    print(f"   ✓ {nrows:,} registros cargados en tabla temporal")
                    
                    # Hacer MERGE
                    merge_pend_query = f"""
                    MERGE INTO {tabla_pendientes} AS target
                    USING {tabla_temp_pend} AS source
                    ON target.AFILIADO_RUT = source.AFILIADO_RUT 
                       AND target.LCC_COMCOR = source.LCC_COMCOR
                    WHEN MATCHED THEN
                        UPDATE SET
                            SEMAFORO = COALESCE(target.SEMAFORO, source.SEMAFORO),
                            PROBABILIDAD_APROBACION = COALESCE(target.PROBABILIDAD_APROBACION, source.PROBABILIDAD_APROBACION),
                            UMBRAL_VERDE = COALESCE(target.UMBRAL_VERDE, source.UMBRAL_VERDE),
                            UMBRAL_AMARILLO = COALESCE(target.UMBRAL_AMARILLO, source.UMBRAL_AMARILLO),
                            CIE_GRUPO = source.CIE_GRUPO,
                            CIE_F = source.CIE_F,
                            DIASSOLICITADO = source.DIASSOLICITADO,
                            FECHA_RECEPCION = source.FECHA_RECEPCION,
                            EPISODIO_ACUM_DIAS = source.EPISODIO_ACUM_DIAS,
                            COT_EDAD = source.COT_EDAD,
                            COT_GENERO = source.COT_GENERO,
                            RENTA_ESTIMADA = source.RENTA_ESTIMADA,
                            TARGET_FT3 = source.TARGET_FT3,
                            TARGET_APRUEBA = source.TARGET_APRUEBA,
                            FECHA_PROCESAMIENTO = COALESCE(target.FECHA_PROCESAMIENTO, source.FECHA_PROCESAMIENTO),
                            FECHA_PROCESAMIENTO_STR = COALESCE(target.FECHA_PROCESAMIENTO_STR, source.FECHA_PROCESAMIENTO_STR),
                            LEAK_FT = source.LEAK_FT,
                            LEAK_CAUSALES = source.LEAK_CAUSALES,
                            LEAK_DIASAUTORIZADOS = source.LEAK_DIASAUTORIZADOS,
                            LEAK_GLOSAS = source.LEAK_GLOSAS,
                            LEAK_ESTADOLM = source.LEAK_ESTADOLM,
                            LEAK_FALLO_PE = source.LEAK_FALLO_PE
                    WHEN NOT MATCHED THEN
                        INSERT (AFILIADO_RUT, LCC_COMCOR, SEMAFORO, PROBABILIDAD_APROBACION,
                               UMBRAL_VERDE, UMBRAL_AMARILLO, CIE_GRUPO, CIE_F, DIASSOLICITADO,
                               FECHA_RECEPCION, EPISODIO_ACUM_DIAS,
                               COT_EDAD, COT_GENERO, RENTA_ESTIMADA, TARGET_FT3, TARGET_APRUEBA,
                               FECHA_PROCESAMIENTO, FECHA_PROCESAMIENTO_STR,
                               LEAK_FT, LEAK_CAUSALES, LEAK_DIASAUTORIZADOS,
                               LEAK_GLOSAS, LEAK_ESTADOLM, LEAK_FALLO_PE)
                        VALUES (source.AFILIADO_RUT, source.LCC_COMCOR, source.SEMAFORO, source.PROBABILIDAD_APROBACION,
                               source.UMBRAL_VERDE, source.UMBRAL_AMARILLO, source.CIE_GRUPO, source.CIE_F, source.DIASSOLICITADO,
                               source.FECHA_RECEPCION, source.EPISODIO_ACUM_DIAS,
                               source.COT_EDAD, source.COT_GENERO, source.RENTA_ESTIMADA, source.TARGET_FT3, source.TARGET_APRUEBA,
                               source.FECHA_PROCESAMIENTO, source.FECHA_PROCESAMIENTO_STR,
                               source.LEAK_FT, source.LEAK_CAUSALES, source.LEAK_DIASAUTORIZADOS,
                               source.LEAK_GLOSAS, source.LEAK_ESTADOLM, source.LEAK_FALLO_PE)
                    """
                    
                    cursor.execute(merge_pend_query)
                    merge_pend_count = cursor.rowcount
                    print(f"   ✓ MERGE completado: {merge_pend_count} registros afectados")
                    
                    # Limpiar tabla temporal
                    cursor.execute(f"DROP TABLE IF EXISTS {tabla_temp_pend}")
                    
                    # Obtener total de registros en la tabla
                    cursor.execute(f"SELECT COUNT(*) FROM {tabla_pendientes}")
                    total_registros = cursor.fetchone()[0]
                    print(f"   ✓ Total registros en tabla: {total_registros:,}")
                else:
                    print(f"   ⚠ Error al cargar datos en tabla temporal")
            else:
                print(f"   - No hay licencias pendientes para procesar")
            
            cursor.close()
            
        except Exception as e:
            print(f"   ⚠ Advertencia al guardar en Databricks SQL: {e}")
            print("   Continuando con generación de Excel...")
        
        # 8. GENERAR REPORTE EXCEL
        print("\n8. Generando reporte Excel diario...")
        filename = f'results/reporte_dia_{timestamp}.xlsx'
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Resumen ejecutivo
            resumen = {
                'Métrica': [
                    'INFORMACIÓN GENERAL',
                    'Período analizado',
                    'Total licencias procesadas',
                    '  - Con dictamen',
                    '  - Pendientes',
                    '',
                    'LICENCIAS PENDIENTES - PREDICCIONES',
                    '  - FT1 (ya aprobados)',
                    '  - VERDE (recomendar aprobación)',
                    '  - AMARILLO (revisar)',
                    '  - ROJO (recomendar rechazo)',
                    '',
                    'LICENCIAS PENDIENTES EN VERDE',
                    '  - Total casos',
                    '  - Días solicitados',
                    '',
                    'VALIDACIÓN CON HISTÓRICO',
                    '  - Precisión zona VERDE',
                    '  - Falsos positivos',
                    '',
                    'UMBRALES UTILIZADOS',
                    '  - Verde (aprobación)',
                    '  - Amarillo (revisión)',
                    '  - Rojo (rechazo)'
                ],
                'Valor': [
                    '',
                    f"{date_from} al {date_to}",
                    f"{total_licencias:,}",
                    f"{len(info_con_dictamen):,}",
                    f"{len(info_pendientes):,}",
                    '',
                    '',
                    f"{(info_pendientes['SEMAFORO'] == 'FT1_APROBADO').sum():,}",
                    f"{(info_pendientes['SEMAFORO'] == 'VERDE').sum():,}",
                    f"{(info_pendientes['SEMAFORO'] == 'AMARILLO').sum():,}",
                    f"{(info_pendientes['SEMAFORO'] == 'ROJO').sum():,}",
                    '',
                    '',
                    f"{len(verde_pendientes):,}" if len(verde_pendientes) > 0 else "0",
                    f"{verde_pendientes['DIASSOLICITADO'].sum():,}" if len(verde_pendientes) > 0 else "0",
                    '',
                    '',
                    f"{precision:.1%}" if len(verde_con_dictamen) > 0 else "N/A",
                    f"{fp:,}" if len(verde_con_dictamen) > 0 else "N/A",
                    '',
                    '',
                    f">= {optimal_verde:.2f}",
                    f"{optimal_amarillo:.2f} - {optimal_verde:.2f}",
                    f"< {optimal_amarillo:.2f}"
                ]
            }
            
            pd.DataFrame(resumen).to_excel(writer, sheet_name='Resumen', index=False)
            
            # Todas las pendientes ya ordenadas anteriormente
            info_pendientes_sorted.to_excel(writer, sheet_name='Licencias_Pendientes', index=False)
            
            # Solo pendientes en verde
            if len(verde_pendientes) > 0:
                verde_pendientes_sorted = verde_pendientes.sort_values(
                    'PROBABILIDAD_APROBACION', ascending=False
                )
                verde_pendientes_sorted.to_excel(writer, sheet_name='Pendientes_Verde', index=False)
            
            # Licencias con dictamen (para validación)
            if len(info_con_dictamen) > 0:
                info_con_dictamen_sorted = info_con_dictamen.sort_values(
                    ['SEMAFORO', 'PROBABILIDAD_APROBACION'], 
                    ascending=[True, False]
                )
                info_con_dictamen_sorted.to_excel(writer, sheet_name='Con_Dictamen', index=False)
            
            # Errores del modelo (FP en verde)
            if len(verde_con_dictamen) > 0:
                errores = verde_con_dictamen[verde_con_dictamen['TARGET_FT3'] == 0]
                if len(errores) > 0:
                    errores_sorted = errores.sort_values('PROBABILIDAD_APROBACION', ascending=False)
                    errores_sorted.to_excel(writer, sheet_name='Falsos_Positivos', index=False)
        
        print(f"\n✓ Reporte guardado exitosamente en: {filename}")
        print(f"\n✓ CONFIRMACIÓN: Se procesaron {total_licencias:,} licencias totales")
        print(f"  - Con dictamen: {len(info_con_dictamen):,}")
        print(f"  - Pendientes: {len(info_pendientes):,}")
        print(f"✓ Todas tienen probabilidad y clasificación asignada")
        print(f"\n✓ DATABRICKS SQL: Tablas actualizadas:")
        print(f"  - FT30_PREDICCIONES_DIARIAS: Todas las predicciones del día")
        print(f"  - FT30_LICENCIAS_PENDIENTES: Solo licencias pendientes")
        
        if disconnect_after:
            loader.disconnect()
        return filename

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if loader.conn and (created_loader or disconnect_after):
            loader.disconnect()
        raise

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta
    
    # Parser para argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Aplicar modelo FastTrack 3.0 diariamente a licencias médicas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar día anterior (uso típico para cron diario)
  python FT3_dia.py
  
  # Últimos 7 días
  python FT3_dia.py --ultimos-dias 7
  
  # Últimos 30 días
  python FT3_dia.py --ultimos-dias 30
  
  # Especificar rango de fechas
  python FT3_dia.py --desde 2024-06-01 --hasta 2024-08-04
  
  # Solo un día específico
  python FT3_dia.py --desde 2024-08-05 --hasta 2024-08-05
  
  # Desactivar optimización de umbrales
  python FT3_dia.py --no-optimizar
  
  # Cambiar tasa de reversión COMPIN
  python FT3_dia.py --compin 0.5
        """
    )
    
    # Argumentos de fecha
    parser.add_argument('--desde', '--from', 
                       type=str,
                       help='Fecha inicial (YYYY-MM-DD). Por defecto: día anterior')
    parser.add_argument('--hasta', '--to', 
                       type=str,
                       help='Fecha final (YYYY-MM-DD). Por defecto: día anterior')
    parser.add_argument('--ultimos-dias', 
                       type=int,
                       help='Procesar los últimos N días (sobrescribe --desde)')
    
    # Otros parámetros
    parser.add_argument('--no-optimizar', 
                       action='store_true',
                       help='Desactivar optimización de umbrales')
    parser.add_argument('--compin', 
                       type=float, 
                       default=0.5,
                       help='Tasa de reversión COMPIN (0.0-1.0). Por defecto: 0.5')
    parser.add_argument('--costo-manual', 
                       type=int, 
                       default=5000,
                       help='Costo de revisión manual. Por defecto: 5000')
    
    args = parser.parse_args()
    
    # Determinar fechas
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Si se especifica --ultimos-dias, tiene prioridad
    if args.ultimos_dias:
        date_to = yesterday
        date_from = (datetime.now() - timedelta(days=args.ultimos_dias)).strftime('%Y-%m-%d')
    else:
        # Fecha final
        if args.hasta:
            date_to = args.hasta
        else:
            date_to = yesterday  # Por defecto: día anterior
        
        # Fecha inicial
        if args.desde:
            date_from = args.desde
        else:
            date_from = yesterday  # Por defecto: día anterior (solo un día)
    
    # Validar fechas
    try:
        datetime.strptime(date_from, '%Y-%m-%d')
        datetime.strptime(date_to, '%Y-%m-%d')
    except ValueError:
        print("ERROR: Las fechas deben estar en formato YYYY-MM-DD")
        exit(1)
    
    # Mostrar configuración
    print(f"\nCONFIGURACIÓN:")
    print(f"- Período: {date_from} al {date_to}")
    print(f"- Optimización de umbrales: {'Sí' if not args.no_optimizar else 'No'}")
    print(f"- Tasa reversión COMPIN: {args.compin*100:.0f}%")
    print(f"- Costo revisión manual: ${args.costo_manual:,}")
    
    # Ejecutar con parámetros
    report_file = apply_model_to_all_licenses(
        date_from=date_from,
        date_to=date_to,
        optimize_thresholds=not args.no_optimizar,
        cost_manual_review=args.costo_manual,
        compin_reversion_rate=args.compin
    )
    
    print(f"\n{'='*80}")
    print(f"PROCESO COMPLETADO")
    print(f"Archivo generado: {report_file}")
    print(f"{'='*80}")
