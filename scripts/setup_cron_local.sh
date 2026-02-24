#!/bin/bash

# Script para configurar cron en macOS local
# Ejecutar con: bash setup_cron_local.sh

echo "Configurando cron para FastTrack 3.0..."

# Ruta del proyecto
PROJECT_PATH="/Users/alfil/Mi unidad/FT3/python_model"

# Crear script ejecutor
cat > "$PROJECT_PATH/scripts/run_ft3_dia.sh" << 'EOF'
#!/bin/bash
# Script ejecutor para FT3_dia.py

# Configurar entorno
export PATH="/usr/local/bin:/usr/bin:/bin"
cd "/Users/alfil/Mi unidad/FT3/python_model"

# Activar entorno virtual si existe
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Crear directorio de logs si no existe
mkdir -p logs

# Ejecutar FT3_dia.py
echo "$(date): Iniciando FT3_dia.py" >> logs/cron.log
python FT3_dia.py >> logs/ft3_dia_$(date +%Y%m%d).log 2>&1
echo "$(date): FT3_dia.py completado con código $?" >> logs/cron.log
EOF

# Hacer ejecutable el script
chmod +x "$PROJECT_PATH/scripts/run_ft3_dia.sh"

# Configurar crontab
echo "Agregando entrada a crontab..."

# Backup del crontab actual
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null

# Agregar nueva entrada (evitar duplicados)
(crontab -l 2>/dev/null | grep -v "run_ft3_dia.sh"; echo "30 7 * * 2-5 $PROJECT_PATH/scripts/run_ft3_dia.sh") | crontab -

echo "✅ Configuración completada!"
echo ""
echo "📋 Programación configurada:"
echo "   - FT3_dia.py: Martes a Viernes a las 7:30 AM"
echo ""
echo "Para verificar:"
echo "   crontab -l"
echo ""
echo "Para ver logs:"
echo "   tail -f $PROJECT_PATH/logs/cron.log"
echo ""
echo "Para desactivar:"
echo "   crontab -l | grep -v 'run_ft3_dia.sh' | crontab -"