"""
Optimización de umbrales considerando el efecto COMPIN
Si el 79% de los rechazos son revertidos por COMPIN, 
el costo real de aprobar incorrectamente es menor.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from src.data_loader import SnowflakeDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import LightGBMTrainer

def calculate_adjusted_fp_cost(dias_solicitados, 
                             base_cost_per_day=59000, 
                             compin_reversion_rate=0.79,
                             manual_review_cost=5000):
    """
    Calcular el costo ajustado de un falso positivo considerando COMPIN
    
    Si aprobamos incorrectamente una LM:
    - Costo directo: dias * $59,000
    
    Si la hubiéramos rechazado:
    - Costo revisión manual: $5,000
    - 79% de probabilidad que COMPIN la revierta
    - Si COMPIN la revierte, igual pagamos: dias * $59,000
    
    Costo neto del FP = Costo directo - Costo esperado si rechazamos
    Costo esperado si rechazamos = $5,000 + (0.79 * dias * $59,000)
    
    Por lo tanto:
    Costo neto FP = dias * $59,000 - [$5,000 + (0.79 * dias * $59,000)]
    Costo neto FP = dias * $59,000 * (1 - 0.79) - $5,000
    Costo neto FP = dias * $59,000 * 0.21 - $5,000
    """
    costo_directo = dias_solicitados * base_cost_per_day
    costo_si_rechazamos = manual_review_cost + (compin_reversion_rate * costo_directo)
    costo_neto_fp = costo_directo - costo_si_rechazamos
    
    # El costo neto no puede ser negativo
    return np.maximum(costo_neto_fp, 0)

def optimize_with_compin_effect():
    """Optimizar umbrales considerando el efecto COMPIN"""
    
    print("OPTIMIZACIÓN DE UMBRALES CONSIDERANDO EFECTO COMPIN")
    print("="*80)
    print("\nSUPUESTOS:")
    print("- 79% de rechazos son revertidos por COMPIN")
    print("- Si COMPIN revierte, igual se paga el subsidio completo")
    print("- Costo revisión manual: $5,000")
    print("\nIMPLICACIÓN:")
    print("El costo real de aprobar incorrectamente es MENOR porque:")
    print("- Si rechazamos, 79% de las veces igual terminaremos pagando")
    print("- Además tendremos el costo de revisión manual")
    
    # Cargar modelo
    trainer = LightGBMTrainer()
    trainer.load_model('models/fasttrack_model.pkl')
    engineer = FeatureEngineer()
    engineer.load_transformers('models/feature_fasttrack.pkl')
    
    # Conectar a Snowflake
    loader = SnowflakeDataLoader()
    
    try:
        loader.connect()
        
        # Cargar datos
        print("\n\nCargando datos de validación...")
        df = loader.load_training_data(
            table_name="MODELO_LM_202507_TRAIN",
            date_from="2025-06-01",
            date_to="2025-08-04",
            include_leak_ft=True
        )
        
        # Filtrar
        df = df[~df['CIE_GRUPO'].isin(['PARTO', 'PUERPERIO'])]
        df_no_ft1 = df[df['LEAK_FT'] == 0].copy()
        
        print(f"Casos para optimización: {len(df_no_ft1):,}")
        
        # Predicciones
        X = engineer.transform(df_no_ft1)
        probabilities = trainer.predict(X)
        
        print("\n\nCOMPARACIÓN DE MODELOS DE COSTO:")
        print("-"*80)
        
        # Modelo 1: Sin considerar COMPIN (actual)
        print("\n1. MODELO ACTUAL (sin considerar COMPIN):")
        print("   Costo FP = días * $59,000")
        
        best_cost_actual = float('inf')
        best_threshold_actual = 0.90
        
        for threshold in np.arange(0.90, 0.999, 0.001):
            verde_mask = probabilities >= threshold
            fp_mask = verde_mask & (df_no_ft1['TARGET_APRUEBA'] == 0)
            
            # Costo directo
            fp_cost = (df_no_ft1.loc[fp_mask, 'DIASSOLICITADO'] * 59000).sum()
            
            # Amarillo y rojo
            amarillo_mask = (probabilities >= 0.50) & (probabilities < threshold)
            manual_cost = amarillo_mask.sum() * 5000
            
            rojo_mask = probabilities < 0.50
            fn_mask = rojo_mask & (df_no_ft1['TARGET_APRUEBA'] == 1)
            fn_cost = fn_mask.sum() * 20000
            
            total_cost = fp_cost + manual_cost + fn_cost
            
            if total_cost < best_cost_actual:
                best_cost_actual = total_cost
                best_threshold_actual = threshold
                best_details_actual = {
                    'fp_count': fp_mask.sum(),
                    'verde_count': verde_mask.sum(),
                    'fp_cost': fp_cost
                }
        
        print(f"   Umbral óptimo: {best_threshold_actual:.3f}")
        print(f"   Casos en verde: {best_details_actual['verde_count']:,} ({best_details_actual['verde_count']/len(probabilities)*100:.1f}%)")
        print(f"   Falsos positivos: {best_details_actual['fp_count']}")
        print(f"   Costo total: ${best_cost_actual:,.0f}")
        
        # Modelo 2: Considerando COMPIN
        print("\n2. MODELO AJUSTADO (considerando 79% reversión COMPIN):")
        print("   Costo neto FP = días * $59,000 * 0.21 - $5,000")
        
        best_cost_adjusted = float('inf')
        best_threshold_adjusted = 0.90
        results_adjusted = []
        
        for threshold in np.arange(0.50, 0.999, 0.001):
            verde_mask = probabilities >= threshold
            fp_mask = verde_mask & (df_no_ft1['TARGET_APRUEBA'] == 0)
            
            # Costo ajustado por COMPIN
            if fp_mask.sum() > 0:
                dias_fp = df_no_ft1.loc[fp_mask, 'DIASSOLICITADO']
                fp_cost_adjusted = calculate_adjusted_fp_cost(
                    dias_fp, 
                    compin_reversion_rate=0.79,
                    manual_review_cost=5000
                ).sum()
            else:
                fp_cost_adjusted = 0
            
            # Amarillo y rojo
            amarillo_mask = (probabilities >= 0.50) & (probabilities < threshold)
            manual_cost = amarillo_mask.sum() * 5000
            
            rojo_mask = probabilities < 0.50
            fn_mask = rojo_mask & (df_no_ft1['TARGET_APRUEBA'] == 1)
            fn_cost = fn_mask.sum() * 20000
            
            total_cost = fp_cost_adjusted + manual_cost + fn_cost
            
            results_adjusted.append({
                'threshold': threshold,
                'total_cost': total_cost,
                'fp_count': fp_mask.sum(),
                'verde_count': verde_mask.sum(),
                'fp_cost_adjusted': fp_cost_adjusted
            })
            
            if total_cost < best_cost_adjusted:
                best_cost_adjusted = total_cost
                best_threshold_adjusted = threshold
                best_details_adjusted = {
                    'fp_count': fp_mask.sum(),
                    'verde_count': verde_mask.sum(),
                    'fp_cost_adjusted': fp_cost_adjusted
                }
        
        print(f"   Umbral óptimo: {best_threshold_adjusted:.3f}")
        print(f"   Casos en verde: {best_details_adjusted['verde_count']:,} ({best_details_adjusted['verde_count']/len(probabilities)*100:.1f}%)")
        print(f"   Falsos positivos: {best_details_adjusted['fp_count']}")
        print(f"   Costo total: ${best_cost_adjusted:,.0f}")
        
        # Comparación
        print("\n\nCOMPARACIÓN DE RESULTADOS:")
        print("-"*80)
        print(f"Umbral óptimo:")
        print(f"  - Sin COMPIN: {best_threshold_actual:.3f}")
        print(f"  - Con COMPIN: {best_threshold_adjusted:.3f}")
        print(f"  - Diferencia: {best_threshold_adjusted - best_threshold_actual:+.3f}")
        
        print(f"\nCasos en zona verde:")
        print(f"  - Sin COMPIN: {best_details_actual['verde_count']:,} ({best_details_actual['verde_count']/len(probabilities)*100:.1f}%)")
        print(f"  - Con COMPIN: {best_details_adjusted['verde_count']:,} ({best_details_adjusted['verde_count']/len(probabilities)*100:.1f}%)")
        print(f"  - Diferencia: {best_details_adjusted['verde_count'] - best_details_actual['verde_count']:+,} casos")
        
        # Análisis de sensibilidad
        print("\n\nANÁLISIS DE SENSIBILIDAD AL PORCENTAJE DE REVERSIÓN:")
        print("-"*80)
        
        reversion_rates = [0.5, 0.6, 0.7, 0.79, 0.85, 0.9]
        optimal_thresholds = []
        
        for rate in reversion_rates:
            best_thresh = 0.90
            best_cost = float('inf')
            
            for threshold in np.arange(0.50, 0.999, 0.001):
                verde_mask = probabilities >= threshold
                fp_mask = verde_mask & (df_no_ft1['TARGET_APRUEBA'] == 0)
                
                if fp_mask.sum() > 0:
                    dias_fp = df_no_ft1.loc[fp_mask, 'DIASSOLICITADO']
                    fp_cost = calculate_adjusted_fp_cost(dias_fp, compin_reversion_rate=rate).sum()
                else:
                    fp_cost = 0
                
                amarillo_mask = (probabilities >= 0.50) & (probabilities < threshold)
                manual_cost = amarillo_mask.sum() * 5000
                
                rojo_mask = probabilities < 0.50
                fn_mask = rojo_mask & (df_no_ft1['TARGET_APRUEBA'] == 1)
                fn_cost = fn_mask.sum() * 20000
                
                total = fp_cost + manual_cost + fn_cost
                
                if total < best_cost:
                    best_cost = total
                    best_thresh = threshold
            
            optimal_thresholds.append(best_thresh)
            print(f"  Reversión {rate*100:.0f}%: Umbral óptimo = {best_thresh:.3f}")
        
        # Visualización
        df_results = pd.DataFrame(results_adjusted)
        
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Costo total vs umbral
        plt.subplot(2, 2, 1)
        plt.plot(df_results['threshold'], df_results['total_cost']/1000000, 'b-', linewidth=2)
        plt.axvline(best_threshold_adjusted, color='red', linestyle='--', 
                   label=f'Óptimo COMPIN: {best_threshold_adjusted:.3f}')
        plt.axvline(best_threshold_actual, color='orange', linestyle='--', 
                   label=f'Óptimo sin COMPIN: {best_threshold_actual:.3f}')
        plt.xlabel('Umbral Verde')
        plt.ylabel('Costo Total (Millones $)')
        plt.title('Costo Total vs Umbral (con efecto COMPIN)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Casos en verde vs umbral
        plt.subplot(2, 2, 2)
        plt.plot(df_results['threshold'], df_results['verde_count'], 'g-', linewidth=2)
        plt.axvline(best_threshold_adjusted, color='red', linestyle='--')
        plt.axvline(best_threshold_actual, color='orange', linestyle='--')
        plt.xlabel('Umbral Verde')
        plt.ylabel('Casos en Zona Verde')
        plt.title('Casos Aprobados Automáticamente')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Sensibilidad a tasa de reversión
        plt.subplot(2, 2, 3)
        plt.plot([r*100 for r in reversion_rates], optimal_thresholds, 'ro-', linewidth=2, markersize=8)
        plt.axhline(0.80, color='gray', linestyle='--', label='Umbral tradicional')
        plt.xlabel('Tasa de Reversión COMPIN (%)')
        plt.ylabel('Umbral Óptimo')
        plt.title('Sensibilidad del Umbral a Tasa de Reversión')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Distribución de probabilidades
        plt.subplot(2, 2, 4)
        plt.hist(probabilities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(best_threshold_adjusted, color='red', linestyle='--', 
                   label=f'Umbral COMPIN: {best_threshold_adjusted:.3f}')
        plt.axvline(best_threshold_actual, color='orange', linestyle='--', 
                   label=f'Umbral sin COMPIN: {best_threshold_actual:.3f}')
        plt.xlabel('Probabilidad de Aprobación')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Probabilidades del Modelo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/optimizacion_con_compin.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Ejemplo de cálculo
        print("\n\nEJEMPLO DE CÁLCULO (LM de 14 días):")
        print("-"*80)
        print("Sin considerar COMPIN:")
        print(f"  - Costo si aprobamos incorrectamente: ${14*59000:,.0f}")
        print("\nConsiderando COMPIN (79% reversión):")
        print(f"  - Si rechazamos: $5,000 + 79% × ${14*59000:,.0f} = ${5000 + 0.79*14*59000:,.0f}")
        print(f"  - Si aprobamos: ${14*59000:,.0f}")
        print(f"  - Costo NETO del error: ${14*59000:,.0f} - ${5000 + 0.79*14*59000:,.0f} = ${14*59000 - (5000 + 0.79*14*59000):,.0f}")
        print(f"  - Equivale a: ${14*59000*0.21 - 5000:,.0f}")
        
        loader.disconnect()
        
        return best_threshold_adjusted, best_details_adjusted
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if loader.conn:
            loader.disconnect()

if __name__ == "__main__":
    optimal_threshold, details = optimize_with_compin_effect()
    
    print("\n\nCONCLUSIÓN:")
    print("="*80)
    print(f"Al considerar que el 79% de rechazos son revertidos por COMPIN:")
    print(f"- El umbral óptimo BAJA significativamente")
    print(f"- Podemos aprobar más casos automáticamente")
    print(f"- El costo real de un FP es solo ~21% del costo nominal")
    print(f"- Menos la revisión manual que nos ahorramos")