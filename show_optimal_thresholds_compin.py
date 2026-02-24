"""
Script rápido para mostrar los umbrales óptimos con diferentes tasas de COMPIN
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('src')

from src.data_loader import SnowflakeDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import LightGBMTrainer
from apply_model_to_all_licenses import optimize_thresholds_by_cost

def quick_threshold_comparison():
    """Comparar umbrales con diferentes tasas COMPIN"""
    
    print("COMPARACIÓN DE UMBRALES CON EFECTO COMPIN")
    print("="*80)
    
    # Cargar modelo
    trainer = LightGBMTrainer()
    trainer.load_model('models/fasttrack_model.pkl')
    engineer = FeatureEngineer()
    engineer.load_transformers('models/feature_fasttrack.pkl')
    
    # Conectar a Snowflake
    loader = SnowflakeDataLoader()
    
    try:
        loader.connect()
        
        # Cargar solo una muestra para ser más rápido
        print("\nCargando datos...")
        df = loader.load_training_data(
            table_name="MODELO_LM_202507_TRAIN",
            date_from="2025-07-01",  # Solo julio para ser más rápido
            date_to="2025-07-31",
            include_leak_ft=True
        )
        
        # Filtrar
        df = df[~df['CIE_GRUPO'].isin(['PARTO', 'PUERPERIO'])]
        df_no_ft1 = df[df['LEAK_FT'] == 0].copy()
        
        print(f"Casos para optimización: {len(df_no_ft1):,}")
        
        # Predicciones
        X = engineer.transform(df_no_ft1)
        probabilities = trainer.predict(X)
        true_labels = df_no_ft1['TARGET_APRUEBA'].values
        dias_solicitados = df_no_ft1['DIASSOLICITADO'].values
        
        # Comparar diferentes tasas
        compin_rates = [0.0, 0.5, 0.79, 0.82]
        results = []
        
        print("\n\nOPTIMIZANDO UMBRALES PARA DIFERENTES TASAS COMPIN:")
        print("-"*80)
        
        for rate in compin_rates:
            print(f"\nTasa COMPIN: {rate*100:.0f}%")
            
            optimal_verde, optimal_amarillo, cost_analysis = optimize_thresholds_by_cost(
                probabilities=probabilities,
                true_labels=true_labels,
                dias_solicitados=dias_solicitados,
                cost_fp_per_day=59000,
                cost_fn=20000,
                cost_manual_review=5000,
                compin_reversion_rate=rate
            )
            
            results.append({
                'Tasa_COMPIN_%': rate * 100,
                'Umbral_Verde': optimal_verde,
                'Umbral_Amarillo': optimal_amarillo,
                'Casos_Verde': cost_analysis['verde_cases'],
                'Casos_Verde_%': cost_analysis['verde_cases'] / len(probabilities) * 100,
                'FP_Verde': cost_analysis['verde_fp'],
                'Costo_Total': cost_analysis['total_cost']
            })
        
        # Mostrar tabla comparativa
        df_results = pd.DataFrame(results)
        
        print("\n\nRESUMEN COMPARATIVO:")
        print("="*80)
        print(f"{'Tasa COMPIN':>12} | {'Umbral Verde':>12} | {'Casos Verde':>12} | {'% Verde':>8} | {'FPs':>6} | {'Costo Total':>15}")
        print("-"*80)
        
        for _, row in df_results.iterrows():
            print(f"{row['Tasa_COMPIN_%']:>11.0f}% | {row['Umbral_Verde']:>12.3f} | "
                  f"{row['Casos_Verde']:>12,} | {row['Casos_Verde_%']:>7.1f}% | "
                  f"{row['FP_Verde']:>6} | ${row['Costo_Total']:>14,.0f}")
        
        # Análisis del impacto
        base_row = df_results[df_results['Tasa_COMPIN_%'] == 0].iloc[0]
        compin_50_row = df_results[df_results['Tasa_COMPIN_%'] == 50].iloc[0]
        
        print("\n\nIMPACTO DE USAR 50% COMPIN (vs no considerarlo):")
        print("-"*80)
        print(f"Umbral verde: {base_row['Umbral_Verde']:.3f} → {compin_50_row['Umbral_Verde']:.3f} "
              f"(cambio: {compin_50_row['Umbral_Verde'] - base_row['Umbral_Verde']:+.3f})")
        print(f"Casos en verde: {base_row['Casos_Verde']:,} → {compin_50_row['Casos_Verde']:,} "
              f"(+{compin_50_row['Casos_Verde'] - base_row['Casos_Verde']:,} casos)")
        print(f"Porcentaje verde: {base_row['Casos_Verde_%']:.1f}% → {compin_50_row['Casos_Verde_%']:.1f}% "
              f"(+{compin_50_row['Casos_Verde_%'] - base_row['Casos_Verde_%']:.1f} puntos)")
        print(f"Ahorro en costos: ${base_row['Costo_Total'] - compin_50_row['Costo_Total']:,.0f}")
        
        loader.disconnect()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if loader.conn:
            loader.disconnect()

if __name__ == "__main__":
    quick_threshold_comparison()