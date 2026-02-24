#!/bin/bash
# Script para actualización mensual de variables
# Se ejecuta el primer día de cada mes a las 3:00 AM

LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/monthly_update_$(date +%Y%m).log"

echo "=========================================" >> $LOG_FILE
echo "Iniciando actualización mensual de variables" >> $LOG_FILE
echo "Fecha: $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

# Ejecutar SQLs de actualización en Snowflake
echo "Ejecutando MODELO_LM.sql..." >> $LOG_FILE
python -c "
import snowflake.connector
import os

conn = snowflake.connector.connect(
    user=os.environ.get('SF_USER'),
    password=os.environ.get('SF_PASSWORD'),
    account=os.environ.get('SF_ACCOUNT'),
    warehouse=os.environ.get('SF_WAREHOUSE'),
    database=os.environ.get('SF_DATABASE'),
    schema=os.environ.get('SF_SCHEMA'),
    role=os.environ.get('SF_ROLE')
)

with open('/app/sql/MODELO_LM.sql', 'r') as f:
    sql = f.read()
    
cursor = conn.cursor()
try:
    cursor.execute(sql)
    print('✓ MODELO_LM.sql ejecutado exitosamente')
except Exception as e:
    print(f'✗ Error ejecutando MODELO_LM.sql: {e}')
    exit(1)
finally:
    cursor.close()
    conn.close()
" >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "Ejecutando MODELO_TRAIN.sql..." >> $LOG_FILE
    python -c "
import snowflake.connector
import os

conn = snowflake.connector.connect(
    user=os.environ.get('SF_USER'),
    password=os.environ.get('SF_PASSWORD'),
    account=os.environ.get('SF_ACCOUNT'),
    warehouse=os.environ.get('SF_WAREHOUSE'),
    database=os.environ.get('SF_DATABASE'),
    schema=os.environ.get('SF_SCHEMA'),
    role=os.environ.get('SF_ROLE')
)

with open('/app/sql/MODELO_TRAIN.sql', 'r') as f:
    sql = f.read()
    
cursor = conn.cursor()
try:
    cursor.execute(sql)
    print('✓ MODELO_TRAIN.sql ejecutado exitosamente')
except Exception as e:
    print(f'✗ Error ejecutando MODELO_TRAIN.sql: {e}')
    exit(1)
finally:
    cursor.close()
    conn.close()
" >> $LOG_FILE 2>&1
fi

if [ $? -eq 0 ]; then
    echo "✓ Actualización mensual completada" >> $LOG_FILE
    
    # Notificar éxito
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ Actualización mensual de variables FT3 completada\"}" \
            $SLACK_WEBHOOK
    fi
else
    echo "✗ Error en actualización mensual" >> $LOG_FILE
    
    # Notificar error
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"❌ Error en actualización mensual FT3\"}" \
            $SLACK_WEBHOOK
    fi
fi

echo "=========================================" >> $LOG_FILE