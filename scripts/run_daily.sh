#!/bin/bash
# Script para ejecución diaria de FT30.py
# Se ejecuta todos los días a las 7:30 AM (después de actualización de LM a las 7:00 AM)

LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/ft30_$(date +%Y%m%d).log"

echo "=========================================" >> $LOG_FILE
echo "Iniciando proceso diario FT30" >> $LOG_FILE
echo "Fecha: $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

# Ejecutar FT30 para procesar licencias del día anterior
python /app/FT30.py --ultimos-dias 1 --no-optimizar >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Proceso completado exitosamente" >> $LOG_FILE
    
    # Opcional: Enviar notificación de éxito
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ FT30 Diario completado: $(date)\"}" \
            $SLACK_WEBHOOK
    fi
else
    echo "✗ Error en el proceso" >> $LOG_FILE
    
    # Opcional: Enviar alerta de error
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"❌ Error en FT30 Diario: $(date)\"}" \
            $SLACK_WEBHOOK
    fi
fi

echo "=========================================" >> $LOG_FILE