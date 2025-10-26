#!/bin/bash

# ==============================================================================
# Description: Starts the Streamlit application - Fuel Analytics
# location: FuelOptiMine/mlflow_server/start_streamlit.sh
# 
# Use:
#   ./start_streamlit.sh           # Modo PRODUCCIÓN (por defecto)
#   ./start_streamlit.sh --dev     # Modo DESARROLLO (con venv)
# ==============================================================================

set -e

# colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Iniciando Streamlit - Fuel Analytics${NC}"
echo -e "${GREEN}================================================${NC}"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ruta del archivo .env
ENV_FILE="$PROJECT_ROOT/.env"

echo -e "${YELLOW}📁 Directorio del script:${NC} $SCRIPT_DIR"
echo -e "${YELLOW}🏠 Raíz del proyecto:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}📄 Archivo .env:${NC} $ENV_FILE"

# Actualizar solo PROJECT_ROOT en .env sin borrar otras variables
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}📋 Archivo .env existente encontrado, actualizando PROJECT_ROOT...${NC}"
    
    # Crear archivo temporal
    TEMP_FILE=$(mktemp)
    
    # Actualizar PROJECT_ROOT si existe, o agregarlo al final
    if grep -q "^PROJECT_ROOT=" "$ENV_FILE"; then
        # Reemplazar línea existente
        sed "s|^PROJECT_ROOT=.*|PROJECT_ROOT=$PROJECT_ROOT|" "$ENV_FILE" > "$TEMP_FILE"
    else
        # Agregar nueva línea al final
        cp "$ENV_FILE" "$TEMP_FILE"
        echo "PROJECT_ROOT=$PROJECT_ROOT" >> "$TEMP_FILE"
    fi
    
    # Reemplazar archivo original
    mv "$TEMP_FILE" "$ENV_FILE"
    echo -e "${GREEN}✓ PROJECT_ROOT actualizado en .env${NC}"
else
    echo -e "${YELLOW}📋 Creando nuevo archivo .env...${NC}"
    echo "PROJECT_ROOT=$PROJECT_ROOT" > "$ENV_FILE"
    echo -e "${GREEN}✓ Archivo .env creado${NC}"
fi

# Mostrar contenido actual del .env
echo -e "${BLUE}📋 Contenido actual del .env:${NC}"
cat "$ENV_FILE"
echo ""

# Determine execution mode (default: production)
if [ "$1" == "--dev" ]; then
    MODE="dev"
    echo -e "${BLUE}🔧 Modo: DESARROLLO (con entorno virtual)${NC}"
else
    MODE="prod"
    echo -e "${BLUE}🔧 Modo: PRODUCCIÓN (sistema)${NC}"
fi

# Verify that the Streamlit file exists
STREAMLIT_APP="$PROJECT_ROOT/frontend/web/app/app.py"
if [ ! -f "$STREAMLIT_APP" ]; then
    echo -e "${RED}❌ Error: No se encontró app.py en $STREAMLIT_APP${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Archivo app.py encontrado${NC}"

# DEVELOPMENT Mode: Activate virtual environment
if [ "$MODE" == "dev" ]; then
    VENV_PATH="$PROJECT_ROOT/fueloptimine-env/bin/activate"
    
    if [ -n "$VIRTUAL_ENV" ]; then
        # Virtual environment is already active
        echo -e "${GREEN}✓ Entorno virtual ya activo: $VIRTUAL_ENV${NC}"
    elif [ -f "$VENV_PATH" ]; then
        # Activate virtual environment
        echo -e "${YELLOW}⚙️  Activando entorno virtual...${NC}"
        source "$VENV_PATH"
        echo -e "${GREEN}✓ Entorno virtual activado${NC}"
    else
        # Virtual environment not found
        echo -e "${RED}❌ Error: No se encontró el entorno virtual en:${NC}"
        echo -e "${RED}   $VENV_PATH${NC}"
        echo -e "${YELLOW}   Cambiando a modo PRODUCCIÓN...${NC}"
        MODE="prod"
    fi
fi

# Verify that Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Error: Streamlit no está instalado${NC}"
    echo -e "${YELLOW}   Instala con: pip install streamlit${NC}"
    exit 1
fi

STREAMLIT_PATH=$(which streamlit)
echo -e "${GREEN}✓ Streamlit encontrado: $STREAMLIT_PATH${NC}"

# Verify Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# Change to project root directory
cd "$PROJECT_ROOT"

# Get network IP to display URL
NETWORK_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "N/A")

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}🚀 Iniciando Streamlit en modo ${MODE^^}${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${YELLOW}📍 URL Local:${NC}    http://localhost:8501"
if [ "$NETWORK_IP" != "N/A" ]; then
    echo -e "${YELLOW}📍 URL Red:${NC}      http://${NETWORK_IP}:8501"
fi
echo ""
echo -e "${YELLOW}💡 Presiona Ctrl+C para detener el servidor${NC}"
echo ""

# run streamlit app
streamlit run frontend/web/app/app.py

# stop message
echo ""
echo -e "${YELLOW}⏹️  Streamlit detenido${NC}"