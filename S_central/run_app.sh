#!/bin/bash
# Exportar las variables de entorno necesarias
export PATH=$PATH:/usr/bin:/bin:/usr/sbin:/sbin
export HOME=/home/server

# Registrar la ejecución del script
echo "Ejecutando script run_app.sh" >> /home/server/SERVER/cron_log.txt

# Ejecutar el script Python
/usr/bin/python3 /home/server/SERVER/app.py >> /home/server/SERVER/app_output.txt 2>&1

