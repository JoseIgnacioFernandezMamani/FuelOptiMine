import sys
from pathlib import Path
import polars as pl
from typing import Tuple, Dict

# Añadir el directorio raíz al path
project_root = Path(__file__).resolve().parents[4]  # Ajustar según estructura real
sys.path.append(str(project_root))

from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.transform.implementation.sensor.transformer import SensorTransformer
from etl_core.extract.config.settings import DATA_DIR

class SensorETLAdapter:
    def __init__(self, truck_id: str = "T-210"):
        self.truck_id = truck_id
        self.metrics = {}
        
    def run_etl(self) -> Tuple[pl.DataFrame, Dict]:
        """Ejecuta el pipeline ETL y retorna DataFrame limpio y métricas"""
        try:
            # 1. Extracción
            extractor = CSVExtractor(
                dataset_name="train_data",
                truck_id=self.truck_id,
                base_data_dir=DATA_DIR
            )
            raw_data, _ = extractor.load_data()
            
            if not raw_data or 'sensor' not in raw_data:
                raise ValueError("Datos de sensor no encontrados")
                
            df_raw = raw_data['sensor']
            
            # 2. Transformación
            transformer = SensorTransformer()
            df_clean = transformer.run_transform(df_raw)
            
            if df_clean.is_empty():
                raise ValueError("DataFrame limpio está vacío")
            
            # 3. Recoger métricas
            self.metrics = {
                'initial_records': transformer.metrics.get('initial_records', 0),
                'cleaned_records': transformer.metrics.get('cleaned_records', 0),
                'clean_percentage': transformer.metrics.get('clean_data_percentage', 0)
            }
            
            return df_clean, self.metrics
            
        except Exception as e:
            raise RuntimeError(f"Error en ETL: {str(e)}") from e