import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Directorio base
    BASE_DIR = Path(__file__).parent.parent
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Logger raíz (configuración base)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Formato común
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d - %(message)s"
    )
    
    # Handlers por módulo (etl_core, analytics, models)
    modules = ["etl_core", "analytics", "models"]
    for module in modules:
        handler = RotatingFileHandler(
            filename=LOGS_DIR / f"{module}.log",
            maxBytes=2*1024*1024,  # 2 MB
            backupCount=3,
            encoding="utf-8"
        )
        handler.setFormatter(formatter)
        
        # Crea logger específico y asigna handler
        logger = logging.getLogger(f"proyecto.{module}")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Nivel detallado para cada módulo
    
    # Handler para consola (solo mensajes importantes)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)