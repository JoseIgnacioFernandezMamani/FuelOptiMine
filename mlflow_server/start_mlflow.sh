#!/bin/bash

# Activar tu entorno virtual
source /mnt/d/Develop/FuelOptiMine/fueloptimine-env/bin/activate

# Rutas absolutas
PROJECT_PATH="/mnt/d/Develop/FuelOptiMine/mlflow_server"
BACKEND_URI="sqlite:///${PROJECT_PATH}/backend/mlflow.db"
ARTIFACT_ROOT="file://${PROJECT_PATH}/artifacts"

# Iniciar MLflow
mlflow server \
    --backend-store-uri $BACKEND_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000 \
    > ${PROJECT_PATH}/logs/mlflow.log 2>&1 &

echo $! > ${PROJECT_PATH}/logs/mlflow.pid
echo "MLflow iniciado en http://localhost:5000"
echo "PID guardado en mlflow_server/logs/mlflow.pid"
