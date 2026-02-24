"""
Análisis detallado de la optimización de umbrales para entender
por qué el modelo encuentra umbrales tan altos (0.98+)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_threshold_optimization():
    """Analizar por qué los umbrales óptimos son tan altos"""
    
    print("ANÁLISIS DE UMBRALES ÓPTIMOS")
    print("="*60)
    
    # Parámetros de costo
    cost_fp_per_day = 59000  # Costo por día de falso positivo
    cost_fn = 20000          # Costo de falso negativo
    cost_manual = 5000       # Costo de revisión manual
    
    print(f"\nESTRUCTURA DE COSTOS:")
    print(f"- Falso Positivo: ${cost_fp_per_day:,} por día")
    print(f"- Falso Negativo: ${cost_fn:,}")
    print(f"- Revisión Manual: ${cost_manual:,}")
    
    # Simular datos realistas
    np.random.seed(42)
    n_samples = 10000
    
    # Generar probabilidades con distribución realista
    # La mayoría de casos tienen probabilidades altas o bajas
    probs = np.concatenate([
        np.random.beta(9, 1, int(n_samples * 0.6)),  # 60% alta probabilidad
        np.random.beta(1, 9, int(n_samples * 0.4))   # 40% baja probabilidad
    ])
    
    # Etiquetas verdaderas correlacionadas con probabilidades
    true_labels = (probs + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # Días solicitados (distribución realista)
    dias = np.random.choice([7, 14, 21, 30], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Evaluar diferentes umbrales
    thresholds = np.arange(0.50, 0.999, 0.001)
    costs = []
    
    for verde_thresh in thresholds:
        # Clasificar casos
        verde_mask = probs >= verde_thresh
        rojo_mask = probs < 0.50
        amarillo_mask = (probs >= 0.50) & (probs < verde_thresh)
        
        # Calcular costos
        # Falsos positivos en verde (aprobar incorrectamente)
        fp_verde = verde_mask & (true_labels == 0)
        fp_cost = (dias[fp_verde] * cost_fp_per_day).sum()
        
        # Falsos negativos en rojo (rechazar incorrectamente)
        fn_rojo = rojo_mask & (true_labels == 1)
        fn_cost = fn_rojo.sum() * cost_fn
        
        # Costo de revisión manual en amarillo
        manual_cost = amarillo_mask.sum() * cost_manual
        
        total_cost = fp_cost + fn_cost + manual_cost
        costs.append({
            'threshold': verde_thresh,
            'total_cost': total_cost,
            'fp_cost': fp_cost,
            'fn_cost': fn_cost,
            'manual_cost': manual_cost,
            'verde_count': verde_mask.sum(),
            'amarillo_count': amarillo_mask.sum(),
            'rojo_count': rojo_mask.sum()
        })
    
    # Convertir a DataFrame
    df_costs = pd.DataFrame(costs)
    
    # Encontrar óptimo
    optimal_idx = df_costs['total_cost'].idxmin()
    optimal = df_costs.iloc[optimal_idx]
    
    print(f"\n\nUMBRAL ÓPTIMO ENCONTRADO: {optimal['threshold']:.3f}")
    print(f"Costo total mínimo: ${optimal['total_cost']:,.0f}")
    
    print(f"\nDESGLOSE DE COSTOS EN EL ÓPTIMO:")
    print(f"- Costo FP: ${optimal['fp_cost']:,.0f} ({optimal['fp_cost']/optimal['total_cost']*100:.1f}%)")
    print(f"- Costo FN: ${optimal['fn_cost']:,.0f} ({optimal['fn_cost']/optimal['total_cost']*100:.1f}%)")
    print(f"- Costo Manual: ${optimal['manual_cost']:,.0f} ({optimal['manual_cost']/optimal['total_cost']*100:.1f}%)")
    
    print(f"\nDISTRIBUCIÓN DE CASOS EN EL ÓPTIMO:")
    print(f"- Verde: {optimal['verde_count']:,} ({optimal['verde_count']/n_samples*100:.1f}%)")
    print(f"- Amarillo: {optimal['amarillo_count']:,} ({optimal['amarillo_count']/n_samples*100:.1f}%)")
    print(f"- Rojo: {optimal['rojo_count']:,} ({optimal['rojo_count']/n_samples*100:.1f}%)")
    
    # Analizar sensibilidad
    print(f"\n\nANÁLISIS DE SENSIBILIDAD:")
    
    # ¿Qué pasa con umbral 0.80?
    idx_80 = np.argmin(np.abs(df_costs['threshold'] - 0.80))
    cost_80 = df_costs.iloc[idx_80]
    
    print(f"\nCon umbral 0.80 (tradicional):")
    print(f"- Costo total: ${cost_80['total_cost']:,.0f}")
    print(f"- Diferencia vs óptimo: +${cost_80['total_cost'] - optimal['total_cost']:,.0f}")
    print(f"- Casos en verde: {cost_80['verde_count']:,} vs {optimal['verde_count']:,} (óptimo)")
    
    # Calcular número de FPs
    verde_80 = probs >= 0.80
    fp_80 = (verde_80 & (true_labels == 0)).sum()
    
    verde_opt = probs >= optimal['threshold']
    fp_opt = (verde_opt & (true_labels == 0)).sum()
    
    print(f"\nFALSOS POSITIVOS:")
    print(f"- Con umbral 0.80: {fp_80} casos")
    print(f"- Con umbral {optimal['threshold']:.3f}: {fp_opt} casos")
    print(f"- Reducción: {fp_80 - fp_opt} casos menos")
    
    # Calcular impacto económico de un solo FP
    avg_dias = dias.mean()
    print(f"\n\nIMPACTO DE UN SOLO FALSO POSITIVO:")
    print(f"- Días promedio: {avg_dias:.1f}")
    print(f"- Costo promedio por FP: ${avg_dias * cost_fp_per_day:,.0f}")
    print(f"- Equivale a: {(avg_dias * cost_fp_per_day) / cost_manual:.0f} revisiones manuales")
    print(f"- Equivale a: {(avg_dias * cost_fp_per_day) / cost_fn:.1f} falsos negativos")
    
    # Visualizar
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_costs['threshold'], df_costs['total_cost'] / 1000000, 'b-', linewidth=2)
    plt.axvline(optimal['threshold'], color='red', linestyle='--', label=f'Óptimo: {optimal["threshold"]:.3f}')
    plt.axvline(0.80, color='orange', linestyle='--', label='Tradicional: 0.80')
    plt.xlabel('Umbral Verde')
    plt.ylabel('Costo Total (Millones $)')
    plt.title('Costo Total vs Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df_costs['threshold'], df_costs['fp_cost'] / 1000000, 'r-', label='Costo FP')
    plt.plot(df_costs['threshold'], df_costs['fn_cost'] / 1000000, 'g-', label='Costo FN')
    plt.plot(df_costs['threshold'], df_costs['manual_cost'] / 1000000, 'b-', label='Costo Manual')
    plt.xlabel('Umbral Verde')
    plt.ylabel('Costo (Millones $)')
    plt.title('Componentes del Costo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(df_costs['threshold'], df_costs['verde_count'] / n_samples * 100, 'g-', label='Verde')
    plt.plot(df_costs['threshold'], df_costs['amarillo_count'] / n_samples * 100, 'y-', label='Amarillo')
    plt.axvline(optimal['threshold'], color='red', linestyle='--')
    plt.xlabel('Umbral Verde')
    plt.ylabel('% de Casos')
    plt.title('Distribución de Casos por Zona')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Zoom en la región óptima
    mask = (df_costs['threshold'] >= optimal['threshold'] - 0.05) & (df_costs['threshold'] <= optimal['threshold'] + 0.05)
    plt.plot(df_costs.loc[mask, 'threshold'], df_costs.loc[mask, 'total_cost'] / 1000000, 'b-', linewidth=2)
    plt.axvline(optimal['threshold'], color='red', linestyle='--')
    plt.xlabel('Umbral Verde')
    plt.ylabel('Costo Total (Millones $)')
    plt.title(f'Zoom: Región Óptima (±0.05)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/analisis_umbral_alto.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n\n¿POR QUÉ EL UMBRAL ÓPTIMO ES TAN ALTO ({optimal['threshold']:.3f})?")
    print("="*60)
    print(f"1. ASIMETRÍA DE COSTOS EXTREMA:")
    print(f"   - Un FP cuesta {cost_fp_per_day/cost_fn:.0f}x más que un FN por día")
    print(f"   - Con {avg_dias:.0f} días promedio, un FP cuesta ${avg_dias*cost_fp_per_day:,.0f}")
    print(f"   - Eso equivale a {(avg_dias*cost_fp_per_day)/cost_fn:.0f} falsos negativos")
    
    print(f"\n2. MEJOR ESTRATEGIA:")
    print(f"   - Ser MUY conservador en aprobación automática (verde)")
    print(f"   - Enviar casos dudosos a revisión manual (amarillo)")
    print(f"   - La revisión manual es barata comparada con aprobar incorrectamente")
    
    print(f"\n3. TRADE-OFF:")
    print(f"   - Menos casos en verde ({optimal['verde_count']/n_samples*100:.1f}% vs {cost_80['verde_count']/n_samples*100:.1f}%)")
    print(f"   - Más casos en amarillo (revisión manual)")
    print(f"   - Pero MUCHO menor costo total")
    
    return optimal['threshold']

if __name__ == "__main__":
    optimal_threshold = analyze_threshold_optimization()
    print(f"\n\nCONCLUSIÓN FINAL:")
    print(f"El umbral óptimo de {optimal_threshold:.3f} es correcto dado los costos.")
    print(f"Un umbral alto protege contra errores muy costosos (FP).")