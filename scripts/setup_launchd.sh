#!/bin/bash

# Script para configurar launchd en macOS
echo "🚀 Configurando programación automática con launchd..."

PLIST_FILE="/Users/alfil/Mi unidad/FT3/python_model/scripts/com.fasttrack.ft3dia.plist"
DEST_FILE="$HOME/Library/LaunchAgents/com.fasttrack.ft3dia.plist"

# Copiar archivo plist
cp "$PLIST_FILE" "$DEST_FILE"

# Cargar el servicio
launchctl load "$DEST_FILE"

echo "✅ Servicio configurado!"
echo ""
echo "📋 Comandos útiles:"
echo "   Ver estado:     launchctl list | grep fasttrack"
echo "   Detener:        launchctl unload $DEST_FILE"
echo "   Iniciar:        launchctl load $DEST_FILE"
echo "   Ejecutar ahora: launchctl start com.fasttrack.ft3dia"
echo ""
echo "📂 Logs en: /Users/alfil/Mi unidad/FT3/python_model/logs/"