#!/bin/bash
# Entrypoint script para el contenedor Docker

echo "========================================="
echo "FastTrack 3.0 - Iniciando contenedor"
echo "Fecha: $(date)"
echo "========================================="

# Verificar conexión a Snowflake
echo "Verificando conexión a Snowflake..."
python -c "
import snowflake.connector
import os
try:
    conn = snowflake.connector.connect(
        user=os.environ.get('SF_USER'),
        password=os.environ.get('SF_PASSWORD'),
        account=os.environ.get('SF_ACCOUNT'),
        warehouse=os.environ.get('SF_WAREHOUSE'),
        database=os.environ.get('SF_DATABASE'),
        schema=os.environ.get('SF_SCHEMA'),
        role=os.environ.get('SF_ROLE')
    )
    print('✓ Conexión exitosa a Snowflake')
    conn.close()
except Exception as e:
    print(f'✗ Error conectando a Snowflake: {e}')
    exit(1)
"

# Verificar que existan los modelos
if [ ! -f "/app/models/fasttrack_model.pkl" ]; then
    echo "⚠ Modelo no encontrado. Ejecutando entrenamiento inicial..."
    python main.py --mode train
fi

# Ejecutar comando pasado como parámetro
exec "$@"