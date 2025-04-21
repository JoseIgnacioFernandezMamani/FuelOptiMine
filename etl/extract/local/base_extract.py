# region IMPORTS
# ============== DATA MANIPULATION ==============
import pandas as pd  # DataFrame operations
from pandas.api.types import CategoricalDtype

# ============== VISUALIZATION ==============
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============== SYSTEM INTERACTION ==============
import os
from glob import glob
from pathlib import Path  # Mejor manejo de rutas moderno
import psutil

# ============== PARALLEL PROCESSING ==============
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== NOTEBOOK UTILITIES ==============
from ipywidgets import interact, widgets, VBox, HBox, Output
from IPython.display import display, clear_output

# ============== TYPE ANNOTATIONS ==============
from typing import List, Dict, Set, Tuple, Optional, Any, Union

# ============== PERFORMANCE & DEBUGGING ==============
import numpy as np
import warnings
from time import perf_counter
import logging
from functools import partial


# region GLOBAL CONFIGURATION
def configure_environment():
    # Configuración de pandas
    pd.set_option('display.precision', 2)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_colwidth', 50)
    
    # Configuración de para ignorar warnings no criticos para mantener limpio la salida
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
    
    # Configuración básica de logging para mejor visualizacion de errores 
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Verificación de versiones
    environment = {
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__,
        'python_version': os.sys.version.split()[0],
        'working_dir': Path.cwd().resolve(),
        'cpu_count': os.cpu_count()
    }
    
    return environment

# Ejecutar configuración y mostrar resultados del estado de los recursos
env_config = configure_environment()

print(f"Python version: {env_config['python_version']}")
print(f"Pandas version: {env_config['pandas_version']}")
print(f"Working directory: {env_config['working_dir']}")
print(f"Available CPUs: {env_config['cpu_count']}")
print(f"Memory available: {psutil.virtual_memory().available / 1e9:.2f} GB")

# Configuración temática común para Plotly
plotly_template = dict(
    layout=go.Layout(
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Segoe UI")
    )
)


import unittest
import pandas as pd
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from time import perf_counter
import traceback
import plotly.express as px
from datetime import time

# .env
# variables de entorno encontrar la ruta de los datos
DATA_DIR = os.path.abspath(os.path.join("..", "data-set"))

COLUMN_MAPPING = {
    "sensor": [
        'ShiftDate', 'Shift', 'TimeStamp', 'RecordDuration', 'Equipment',
        'TruckFleet', 'FuelLevel', 'FuelLevelLiters', 'FuelGauge', 'Speed',
        'RPM', 'Ralenti', 'Latitude', 'Longitude', 'Elevation'
    ],
    "time_model": [
        'ShiftDate', 'Shift', 'TimeStamp', 'RecordDuration', 'Equipment',
        'TruckFleet', 'Status', 'Category', 'Event'
    ],
    "cycle": [
        'travel_empty_seconds', 'haul_loaded_seconds', 'waiting_loader_seconds',
        'spotting_time_seconds', 'loading_time_seconds', 'waiting_dump_seconds',
        'reversing_seconds', 'tipping_time_seconds', 'total_travel_time_seconds',
        'operational_time_seconds', 'total_cycle_time_seconds', 'loaded_distance_m',
        'empty_distance_m', 'equivalent_distance_m', 'measured_tonnage',
        'reported_tonnage', 'loading_easting', 'loading_northing',
        'loading_elevation_cm', 'loading_slope_percent', 'dump_easting',
        'dump_northing', 'dump_elevation_cm', 'dump_slope_percent',
        'loading_TimeStamp', 'dumping_TimeStamp', 'ShiftDate', 'shift_type',
        'truck_name', 'truck_model', 'fleet_name', 'loading_face', 'material_type',
        'dump_type', 'dump_location', 'TimeStamp'
    ]
}

# config.py
# pip install python-dotenv
""""
implementar en un futuro
import os
import json
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '/ruta/por/defecto')
COLUMN_MAPPING = json.loads(os.getenv('COLUMN_MAPPING', '{"sensor":[], "time_model":[], "cycle":[]}'))
"""
# clase para cargar los datos
# from config import DATA_DIR, COLUMN_MAPPING
class DataLoader:
    def __init__(self, truck: str):

        self.base_dir = DATA_DIR
        self.truck = truck.upper()
        self._validate_truck_exists() 

    # validar que el camion exista
    def _validate_truck_exists(self):
        """Valida que existan archivos para el camión especificado"""
        pattern = os.path.join(self.base_dir, "train_data_*", "test_*", f"{self.truck}_*.csv")
        files = glob(pattern)
        if not files:
            raise ValueError(f"No se encontraron archivos para el camión {self.truck}")

    def _generate_file_patterns(self, truck: str, data_type: str) -> list:
        """Genera patrones de búsqueda para tipo de conjunto y año"""
        return [
            os.path.join(
                self.base_dir,
                f"train_data_{data_type}",
                f"test_*",
                f"{self.truck}_*.csv"
            )
        ]
    
    def _find_matching_files(self, patterns: list) -> list:
        """Encuentra archivos que coincidan con los patrones"""
        return [file for pattern in patterns for file in glob(pattern)]
    
    def _load_single_file(self, file_path: str, data_type: str) -> pd.DataFrame:
        """Carga y procesa un solo archivo CSV"""
        try:
            # Detectar separador
            sep = self._detect_separator(file_path)
            
            # Cargar datos
            df = pd.read_csv(
                file_path,
                sep=sep,
                header=None,
                names=COLUMN_MAPPING[data_type],
                encoding='utf-8-sig',
                on_bad_lines='warn'
            )

            return df

        except Exception as e:
            print(f"Error en {file_path}: {str(e)}")
            return pd.DataFrame()

    def _detect_separator(self, file_path: str) -> str:
        """Detecta el separador leyendo la primera línea"""
        separadores = [';', '\t', ',']
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                primera_linea = f.readline().strip()
            return next((s for s in separadores if s in primera_linea), ',')
        except:
            return ','

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Carga todos los tipos de datos en DataFrames separados"""
        
        datasets = {}
        
        # Iterar sobre cada tipo de datos definido en COLUMN_MAPPING (sensor, time_model, cycle)
        for data_type in COLUMN_MAPPING:
            try:
                patterns = self._generate_file_patterns(self.truck, data_type)
                files = self._find_matching_files(patterns)
                
                if not files:
                    print(f"Advertencia: No se encontraron archivos de {data_type}")
                    datasets[data_type] = pd.DataFrame()
                    continue
                
                with ThreadPoolExecutor() as executor:
                    dfs = list(executor.map(lambda f: self._load_single_file(f, data_type), files))
                
                combined_df = pd.concat(dfs, ignore_index=True)
                datasets[data_type] = combined_df.sort_values('TimeStamp') if 'TimeStamp' in combined_df else combined_df

                ## feedback para ver si se cargaron los datos correctamente
                print(f"feedback {data_type.upper()}: Cargados {len(combined_df)} registros")
            
            except KeyError:
                print(f"Columnas no definidas para {data_type}")
                datasets[data_type] = pd.DataFrame()
        
        return datasets











class DataLoader:
# Configuraciones como constantes de clase
    EXPECTED_COLUMNS = ['name', 'fleet', 'created_at_local', 'value', 'speed']
    VALID_METRICS = {'rpm', 'cycle', 'time_model'}
    VALID_YEARS = {'2025'}
    CSV_SEPARATORS = {';', '\t', ','}  # Soporta múltiples delimitadores
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.csv_columns = ['name', 'fleet', 'created_at_local', 'value', 'speed']  # nombre de columnas de 
        self.available_trucks = self._discover_available_trucks()
        
    def _discover_available_trucks(self) -> list:
        pattern = os.path.join(self.base_dir, 'data-set', 'explore_data_*', 'test_*', '*.csv')
        files = glob(pattern)
        return sorted(list({os.path.basename(f).split('_')[0] for f in files}))
    
    def _validate_inputs(self, truck_name: str, metrics: list, years: list):
        if truck_name not in self.available_trucks:
            raise ValueError(f"Camion {truck_name} no disponible")
        if not all(m in ['fuel', 'rpm', 'cycle', 'time_model'] for m in metrics):
            raise ValueError("Métricas inválidas. Opciones válidas: fuel, rpm, cycle, time_model")
        if not all(str(y) in ['2024', '2025'] for y in years):
            raise ValueError("Años deben ser 2024 o 2025")

    def _generate_file_patterns(self, truck_name: str, metrics: list, years: list) -> list:
        return [
            os.path.join(
                self.base_dir,
                'data-set',
                f'explore_data_{metric}',
                f'test_{year}',
                f'{truck_name}_{metric}_{year}-*.csv'
            )
            for metric in metrics
            for year in years
        ]

    def _find_matching_files(self, patterns: list) -> list:
        return [f for pattern in patterns for f in glob(pattern)]

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        try:
            # Extraer metadatos del nombre del archivo
            filename = os.path.basename(file_path)
            truck, metric, year_part = filename.split('_', 2)
            year = year_part.split('-')[0]

            # Cargar datos del CSV
            df = pd.read_csv(
                file_path,
                sep=';' if ';' in open(file_path).readline() else '\t',  # Detección de separador
                header=None,
                names=self.csv_columns,  # Asignar nombres de columnas
                encoding='utf-8-sig',
                parse_dates=['created_at_local'],
                date_format='%Y-%m-%d %H:%M:%S.%f',  # Formato EXPLÍCITO
                on_bad_lines='warn',
                dtype={'speed': 'Int8'}
            )
            
            # Añadir metadatos como nuevas columnas
            df['truck'] = truck  # viene del filename
            df['metric'] = metric  # viene del filename
            df['year'] = year  # viene del filename
            df['equipo'] = df['fleet']  # si 'equipo' es un alias de 'fleet'

            # Validación básica
            if df['created_at_local'].isnull().all():
                raise ValueError("No se pudieron parsear las fechas")
                
            return df.dropna(subset=['created_at_local', 'value'])

        except Exception as e:
            print(f"Error en {filename}: {str(e)}")
            return pd.DataFrame()

    def load_data(self, truck_name: str, metrics: list, years: list) -> pd.DataFrame:
        self._validate_inputs(truck_name, metrics, years)
        patterns = self._generate_file_patterns(truck_name, metrics, years)
        files = self._find_matching_files(patterns)
        
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para {truck_name}")
            
        # Carga paralelizada
        with ThreadPoolExecutor() as executor:
            dfs = list(executor.map(self._load_single_file, files))
        
        # Consolidación y ordenamiento
        final_df = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
        return final_df.sort_values('created_at_local').reset_index(drop=True)
        