#!/bin/bash

PID_FILE="/mnt/d/Develop/FuelOptiMine/mlflow_server/logs/mlflow.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null; then
        kill $PID
        rm $PID_FILE
        echo "MLflow detenido (PID: $PID)"
    else
        echo "Proceso no encontrado, limpiando PID file"
        rm $PID_FILE
    fi
else
    echo "MLflow no está corriendo"
fi
