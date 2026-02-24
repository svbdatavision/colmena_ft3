#!/bin/bash
# Script para validación semanal del modelo
# Se ejecuta todos los lunes a las 7:00 AM

LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/validation_$(date +%Y%m%d).log"

echo "=========================================" >> $LOG_FILE
echo "Iniciando validación semanal del modelo" >> $LOG_FILE
echo "Fecha: $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

# Ejecutar validación con datos de la última semana
python /app/validate_model.py --ultimos-dias 7 >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Validación completada" >> $LOG_FILE
    
    # Generar reporte de métricas
    python /app/generate_metrics_report.py >> $LOG_FILE 2>&1
    
    # Verificar si el modelo necesita recalibración
    python -c "
import json
import sys

with open('/app/reports/latest_metrics.json', 'r') as f:
    metrics = json.load(f)
    
if metrics['auc'] < 0.7 or metrics['precision'] < 0.6:
    print('⚠ ALERTA: Modelo necesita recalibración')
    print(f'  AUC: {metrics[\"auc\"]:.3f}')
    print(f'  Precisión: {metrics[\"precision\"]:.3f}')
    sys.exit(1)
else:
    print('✓ Modelo dentro de parámetros')
    sys.exit(0)
" >> $LOG_FILE 2>&1
    
    if [ $? -ne 0 ] && [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"⚠️ Modelo FT3 necesita recalibración. Revisar métricas.\"}" \
            $SLACK_WEBHOOK
    fi
else
    echo "✗ Error en validación" >> $LOG_FILE
fi

echo "=========================================" >> $LOG_FILE