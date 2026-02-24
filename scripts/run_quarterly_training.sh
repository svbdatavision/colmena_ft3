#!/bin/bash
# Script para reentrenamiento trimestral del modelo
# Se ejecuta el primer día de cada trimestre (Enero, Abril, Julio, Octubre) a las 2:00 AM

LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d).log"
BACKUP_DIR="/app/models/backup"

echo "=========================================" >> $LOG_FILE
echo "Iniciando reentrenamiento trimestral" >> $LOG_FILE
echo "Fecha: $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

# Crear backup del modelo actual
mkdir -p $BACKUP_DIR
if [ -f "/app/models/fasttrack_model.pkl" ]; then
    echo "Creando backup del modelo actual..." >> $LOG_FILE
    cp /app/models/fasttrack_model.pkl "$BACKUP_DIR/fasttrack_model_$(date +%Y%m%d).pkl"
    cp /app/models/feature_fasttrack.pkl "$BACKUP_DIR/feature_fasttrack_$(date +%Y%m%d).pkl"
fi

# Entrenar nuevo modelo
echo "Entrenando nuevo modelo..." >> $LOG_FILE
python /app/main.py --mode train >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Modelo entrenado exitosamente" >> $LOG_FILE
    
    # Validar el nuevo modelo
    echo "Validando nuevo modelo..." >> $LOG_FILE
    python /app/validate_model.py --validate-new >> $LOG_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Modelo validado exitosamente" >> $LOG_FILE
        
        # Generar reporte de entrenamiento
        python /app/generate_training_report.py >> $LOG_FILE 2>&1
        
        # Notificar éxito
        if [ ! -z "$SLACK_WEBHOOK" ]; then
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"✅ Reentrenamiento trimestral FT3 completado exitosamente\"}" \
                $SLACK_WEBHOOK
        fi
    else
        echo "✗ Validación falló, restaurando modelo anterior" >> $LOG_FILE
        
        # Restaurar modelo anterior
        cp "$BACKUP_DIR/fasttrack_model_$(date +%Y%m%d).pkl" /app/models/fasttrack_model.pkl
        cp "$BACKUP_DIR/feature_fasttrack_$(date +%Y%m%d).pkl" /app/models/feature_fasttrack.pkl
        
        # Notificar fallo
        if [ ! -z "$SLACK_WEBHOOK" ]; then
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"❌ Reentrenamiento FT3 falló. Modelo anterior restaurado.\"}" \
                $SLACK_WEBHOOK
        fi
    fi
else
    echo "✗ Error en entrenamiento" >> $LOG_FILE
    
    # Notificar error
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"❌ Error crítico en reentrenamiento FT3\"}" \
            $SLACK_WEBHOOK
    fi
fi

echo "=========================================" >> $LOG_FILE