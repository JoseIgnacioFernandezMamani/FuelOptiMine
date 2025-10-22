#!/bin/bash

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MLFLOW_LOGS_PATH="${PROJECT_ROOT}/mlflow_server/logs"
PID_FILE="${MLFLOW_LOGS_PATH}/mlflow.pid"

# Función para matar el proceso y todos sus hijos
kill_process_tree() {
    local pid=$1
    local sig=${2:-TERM}
    
    # Obtener todos los PIDs hijos
    local children=$(pgrep -P $pid 2>/dev/null || true)
    
    # Matar proceso principal
    kill -$sig $pid 2>/dev/null || true
    
    # Matar todos los hijos
    for child in $children; do
        kill_process_tree $child $sig
    done
}

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${YELLOW}🛑 Deteniendo MLflow (PID: $PID y procesos hijos)...${NC}"
        
        # Intentar detención suave
        kill_process_tree $PID TERM
        
        sleep 2
        
        # Si aún está corriendo, forzar detención
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  Forzando detención...${NC}"
            kill_process_tree $PID KILL
        fi
        
        rm "$PID_FILE"
        echo -e "${GREEN}✅ MLflow detenido${NC}"
    else
        echo -e "${YELLOW}⚠️  Proceso no encontrado, limpiando PID${NC}"
        rm "$PID_FILE"
    fi
else
    echo -e "${RED}❌ MLflow no está corriendo${NC}"
fi

# Verificación adicional: matar cualquier proceso mlflow residual
ORPHAN_PIDS=$(pgrep -f "mlflow server" 2>/dev/null || true)
if [ -n "$ORPHAN_PIDS" ]; then
    echo -e "${YELLOW}⚠️  Detectados procesos huérfanos de MLflow, eliminando...${NC}"
    echo $ORPHAN_PIDS | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}✅ Procesos huérfanos eliminados${NC}"
fi