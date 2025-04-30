import os
import pandas as pd
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from pathlib import Path
import traceback

from extract.interfaces.base import IBaseExtractor
from extract.models.schemas import COLUMN_MAPPING
from extract.config.settings import DATA_DIR

class ILocalExtractor(IBaseExtractor):
    def __init__(self, truck: str):
        self.truck = truck.upper()
        self.base_dir = DATA_DIR
        self._validate_truck_exists()

    def _validate_truck_exists(self) -> None:
        """Valida la existencia de archivos para el camión"""
        pattern = self.base_dir / "train_data_*" / "test_*" / f"{self.truck}_*.csv"
        files = glob(str(pattern))
        if not files:
            raise FileNotFoundError(f"No hay datos para {self.truck}")

    def _generate_file_patterns(self, data_type: str) -> List[Path]:
        """Genera patrones de búsqueda para el tipo de datos"""
        return [
            self.base_dir / f"train_data_{data_type}" / "test_*" / f"{self.truck}_*.csv"
        ]

    @staticmethod
    def _detect_separator(file_path: Path) -> str:
        """Detecta el separador del CSV"""
        separators = [';', '\t', ',']
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip()
            return next((s for s in separators if s in first_line), ',')
        except Exception as e:
            print(f"Error detectando separador: {str(e)}")
            return ','

    def _load_single_file(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Carga y procesa un archivo individual"""
        try:
            sep = self._detect_separator(file_path)
            return pd.read_csv(
                file_path,
                sep=sep,
                header=None,
                names=COLUMN_MAPPING[data_type],
                encoding='utf-8-sig',
                on_bad_lines='warn'
            )
        except Exception as e:
            print(f"Error cargando {file_path.name}: {str(e)}")
            return pd.DataFrame()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Carga todos los tipos de datos"""
        datasets = {}
        
        for data_type in COLUMN_MAPPING:
            try:
                patterns = self._generate_file_patterns(data_type)
                files = [f for p in patterns for f in glob(str(p))]
                
                if not files:
                    print(f"Advertencia: Sin archivos para {data_type}")
                    datasets[data_type] = pd.DataFrame()
                    continue
                
                with ThreadPoolExecutor() as executor:
                    dfs = list(executor.map(lambda f: self._load_single_file(Path(f), data_type)))
                
                combined_df = pd.concat(dfs, ignore_index=True)
                if 'TimeStamp' in combined_df:
                    combined_df.sort_values('TimeStamp', inplace=True)
                
                datasets[data_type] = combined_df
                print(f"[{data_type.upper()}] Cargados {len(combined_df)} registros")
            except Exception as e:
                print(f"Error procesando datos para {data_type}: {str(e)}")
                datasets[data_type] = pd.DataFrame()
            
            except KeyError:
                print(f"Esquema no definido para {data_type}")
                datasets[data_type] = pd.DataFrame()
        
        return datasets
