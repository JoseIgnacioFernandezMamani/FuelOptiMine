#!/bin/bash

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo -e "${RED}❌ Archivo .env no encontrado${NC}"
    exit 1
fi

MLFLOW_SERVER_PATH="${PROJECT_ROOT}/mlflow_server"
MLFLOW_LOGS_PATH="${MLFLOW_SERVER_PATH}/logs"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_SERVER_PATH}/artifacts"

mkdir -p "${MLFLOW_LOGS_PATH}"
mkdir -p "${MLFLOW_ARTIFACT_ROOT}"

BACKEND_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
ARTIFACT_ROOT="file://${MLFLOW_ARTIFACT_ROOT}"

PID_FILE="${MLFLOW_LOGS_PATH}/mlflow.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  MLflow ya está corriendo (PID: $OLD_PID)${NC}"
        exit 1
    fi
    rm "$PID_FILE"
fi

echo -e "${GREEN}🚀 Iniciando MLflow...${NC}"

mlflow server \
    --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" \
    > "${MLFLOW_LOGS_PATH}/mlflow.log" 2>&1 &

MLFLOW_PID=$!
echo $MLFLOW_PID > "$PID_FILE"

if ! ps -p $MLFLOW_PID > /dev/null 2>&1; then
    echo -e "${RED}❌ Error al iniciar${NC}"
    tail -n 20 "${MLFLOW_LOGS_PATH}/mlflow.log"
    exit 1
fi

check_mlflow_health() {
    curl -s "http://localhost:${MLFLOW_PORT}/health" > /dev/null 2>&1
}

MAX_ATTEMPTS=300
ATTEMPT=0
SPINNER=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
SPINNER_IDX=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if ! ps -p $MLFLOW_PID > /dev/null 2>&1; then
        echo -e "\n${RED}❌ Proceso detenido inesperadamente${NC}"
        tail -n 20 "${MLFLOW_LOGS_PATH}/mlflow.log"
        exit 1
    fi
    
    if check_mlflow_health; then
        break
    fi
    
    printf "\r${YELLOW}${SPINNER[$SPINNER_IDX]} Esperando... (%ds)${NC}" $ATTEMPT
    SPINNER_IDX=$(( (SPINNER_IDX + 1) % ${#SPINNER[@]} ))
    
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
    echo -e "\n${RED}❌ Timeout después de ${MAX_ATTEMPTS}s${NC}"
    tail -n 20 "${MLFLOW_LOGS_PATH}/mlflow.log"
    exit 1
fi

echo -e "\n${GREEN}✅ MLflow disponible en http://localhost:${MLFLOW_PORT}${NC}"
echo -e "${GREEN}🆔 PID: $MLFLOW_PID${NC}"