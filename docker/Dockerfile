# Usar Python oficial - requiere docker login
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema incluyendo cron y tzdata para manejar la zona horaria
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Santiago
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cron \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para mejor cache
COPY python_model/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación y dependencias adicionales
COPY python_model/ /app/
COPY Variables_cat_train.py /app/

# Asegurar permisos de ejecución en scripts principales
RUN chmod +x /app/run_daily_pipeline.sh \
    /app/run_weekly_pipeline.sh \
    /app/run_monthly_pipeline.sh

# Copiar y configurar crontab
COPY python_model/docker/crontab /etc/cron.d/ft3-cron
RUN chmod 0644 /etc/cron.d/ft3-cron \
    && crontab /etc/cron.d/ft3-cron

# Crear directorio para logs
RUN mkdir -p /app/logs /app/results /app/models

# Variables de entorno (se sobreescriben con docker-compose.yml)
ENV PYTHONUNBUFFERED=1

# Script de inicio que ejecuta cron y mantiene el contenedor activo
RUN cat <<'EOF' > /app/start.sh
#!/bin/bash
touch /var/log/cron.log
service cron start
{
  echo "FastTrack 3.0 Scheduler iniciado en $(date '+%c %Z')"
  echo "Programación real (crontab -l):"
  crontab -l
} >> /var/log/cron.log
tail -f /var/log/cron.log
EOF
RUN chmod +x /app/start.sh

# Comando por defecto
CMD ["/app/start.sh"]
