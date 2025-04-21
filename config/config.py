# config/loader.py
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración desde un archivo YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config