"""
MLflow Configuration Module

Este módulo centraliza toda la configuración de MLflow y PostgreSQL.
Úsalo en tus scripts de entrenamiento para conectarte al tracking server.

Ejemplo de uso:
    from mlflow_server.config import MLFLOW_TRACKING_URI
    import mlflow
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Detectar la raíz del proyecto automáticamente
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cargar variables de entorno
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    print(f"⚠️  Advertencia: No se encontró {ENV_FILE}")
    print("   Usando valores por defecto o variables de entorno del sistema")

# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_HOST = os.getenv("MLFLOW_HOST", "0.0.0.0")
MLFLOW_PORT = int(os.getenv("MLFLOW_PORT", "5000"))

# PostgreSQL
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fuel_optimine")
POSTGRES_USER = os.getenv("POSTGRES_USER", "msc_user2_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# MLflow Backend Store (PostgreSQL)
MLFLOW_BACKEND_STORE_URI = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Paths
MLFLOW_SERVER_PATH = PROJECT_ROOT / "mlflow_server"
MLFLOW_ARTIFACT_ROOT = os.getenv(
    "MLFLOW_ARTIFACT_ROOT",
    str(MLFLOW_SERVER_PATH / "artifacts")
)
MLFLOW_LOGS_PATH = MLFLOW_SERVER_PATH / "logs"

# ============= TRAINING CONFIGURATION =============
TRUCK_IDS = [
    "T-210", "T-211", "T-212", "T-213", "T-214", "T-215", "T-216", "T-217",
    "T-218", "T-219", "T-220", "T-221", "T-222", "T-223", "T-224", "T-225",
    "T-230", "T-231", "T-232", "T-233", "T-236", "T-237", "T-238",
    "T-240", "T-241", "T-242", "T-243"
]

NUMERIC_PREDICTOR_VARS = [
    "SpeedAvg",
    "TotalMeasuredTonnage",
    "Distance",
    "CycleDurationSeconds",
    "StageSequence",
    "TimeEfficiencyPercentage",
]

CATEGORICAL_VARS = ["Destination", "DestinationType", "Material", "Shovel"]

def validate_config():
    """Valida que las configuraciones críticas estén presentes"""
    required_vars = {
        "POSTGRES_HOST": POSTGRES_HOST,
        "POSTGRES_DB": POSTGRES_DB,
        "POSTGRES_USER": POSTGRES_USER,
        "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
    }
    
    missing = [var for var, val in required_vars.items() if not val]
    
    if missing:
        raise ValueError(f"❌ Faltan variables: {', '.join(missing)}")
    
    # Crear directorios
    MLFLOW_LOGS_PATH.mkdir(parents=True, exist_ok=True)
    Path(MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
    
    print("✅ Configuración validada")
    print(f"📁 Proyecto: {PROJECT_ROOT}")
    print(f"📊 Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"🗄️  Backend: PostgreSQL ({POSTGRES_DB})")


def get_mlflow_client():
    """
    Retorna un cliente MLflow configurado
    
    Returns:
        mlflow.tracking.MlflowClient
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


if __name__ == "__main__":
    validate_config()