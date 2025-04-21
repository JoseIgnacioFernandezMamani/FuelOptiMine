from typing import Dict, Union
import pandas as pd

class DataValidator:
    def __init__(self, raw_data: Dict[str, pd.DataFrame], processed_data: Dict[str, pd.DataFrame]):
        """
        Args:
            raw_data: Diccionario con datasets crudos
            processed_data: Diccionario con datasets procesados
        """
        self.raw_data = self._validate_input(raw_data, "raw")
        self.processed_data = self._validate_input(processed_data, "processed")
        self.truck_id = self._detect_truck_id()
        self.required_columns = {
            'sensor': ['ShiftDate', 'TimeStamp', 'Speed', 'RPM', 'FuelLevel'],
            'time_model': ['ShiftDate', 'TimeStamp', 'Status'],
            'cycle': []
        }

    def _validate_input(self, data: dict, data_type: str) -> Dict[str, pd.DataFrame]:
        """Valida estructura de los datos de entrada"""
        if not isinstance(data, dict):
            raise TypeError(f"{data_type} debe ser un diccionario")
        for key, df in data.items():
            if not isinstance(key, str):
                raise TypeError(f"Claves en {data_type} deben ser strings")
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Valores en {data_type} deben ser DataFrames")
        return data

    def _detect_truck_id(self) -> str:
        """Detecta ID del camión desde datos procesados o crudos"""
        for dataset in [self.processed_data, self.raw_data]:
            for dtype in ['sensor', 'time_model', 'cycle']:
                df = dataset.get(dtype, pd.DataFrame())
                if not df.empty and 'Equipment' in df.columns:
                    return df['Equipment'].iloc[0]
        return 'N/A'

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, str]:
        """Calcula métricas para un DataFrame individual"""
        metrics = {
            'filas': '0',
            'columnas': '0',
            'nulos': '0',
            'duplicados': '0',
            'rango_temporal': 'N/A'
        }
        
        if not df.empty:
            metrics.update({
                'filas': f"{len(df):,}",
                'columnas': f"{len(df.columns):,}",
                'nulos': f"{df.isnull().sum().sum():,}",
                'duplicados': f"{df.duplicated().sum():,}"
            })
            
            if 'ShiftDate' in df.columns:
                valid_dates = pd.to_datetime(df['ShiftDate'], errors='coerce').dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min().strftime('%Y-%m-%d')
                    max_date = valid_dates.max().strftime('%Y-%m-%d')
                    metrics['rango_temporal'] = f"{min_date} - {max_date}"
        
        return metrics

    def generate_comparison_table(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Genera tabla comparativa entre datos crudos y procesados"""
        results = {}
        
        for dtype in ['sensor', 'time_model', 'cycle']:
            raw_df = self.raw_data.get(dtype, pd.DataFrame())
            processed_df = self.processed_data.get(dtype, pd.DataFrame())
            
            results[dtype] = {
                'crudo': self._calculate_metrics(raw_df),
                'procesado': self._calculate_metrics(processed_df)
            }
        
        return results

    def print_comparison(self):
        """Imprime tabla comparativa formateada"""
        comparison = self.generate_comparison_table()
        
        print(f"\n{'='*65}")
        print(f" COMPARATIVO DATOS CRUDOS vs PROCESADOS - CAMIÓN: {self.truck_id} ")
        print(f"{'='*65}\n")
        
        # Encabezados
        header = (
            f"{'Dataset':<12} | {'Tipo':<8} | {'Filas':>10} | {'Columnas':>10} | "
            f"{'Nulos':>10} | {'Duplicados':>12} | {'Rango Temporal'}"
        )
        print(header)
        print("-"*95)
        
        # Filas
        for dtype, data in comparison.items():
            print(f"{dtype.upper():<12} | {'CRUDO':<8} | "
                  f"{data['crudo']['filas']:>10} | {data['crudo']['columnas']:>10} | "
                  f"{data['crudo']['nulos']:>10} | {data['crudo']['duplicados']:>12} | "
                  f"{data['crudo']['rango_temporal']}")
            
            print(f"{'':<12} | {'PROCESADO':<8} | "
                  f"{data['procesado']['filas']:>10} | {data['procesado']['columnas']:>10} | "
                  f"{data['procesado']['nulos']:>10} | {data['procesado']['duplicados']:>12} | "
                  f"{data['procesado']['rango_temporal']}")
            print("-"*95)






# Cargar datos originales
loader = DataLoader(DATA_DIR, "T-234")
raw_datasets = loader.load_data()

# Procesar y comparar
etl_processor = ETLDataProcessor(raw_datasets)
etl_processor.run_etl().show_comparison()
for dtype in self.processed_datasets:
    df = self.processed_datasets[dtype]

    # 1. Limpiar columnas no estándar
    df = self._enforce_standard_columns(df, dtype)

    # 2. Limpieza genérica
    if 'TimeStamp' in df.columns:
        df.dropna(subset=['TimeStamp'], inplace=True)

    # 3. Limpieza específica
    if dtype == 'sensor':
        df.dropna(subset=['Speed', 'RPM'], inplace=True)
        df = df[(df['Speed'] > 0) & (df['RPM'] > 0)]

    elif dtype == 'time_model':
        df = df[df['Status'].isin(['active', 'idle'])]

    # 👉 4. ¡GUARDAR LOS CAMBIOS!
    self.processed_datasets[dtype] = df




import pandas as pd
from typing import Dict

class DataValidator:
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
        self.truck_id = self._detect_truck_id()
        self.required_columns = {
            'sensor': ['ShiftDate', 'TimeStamp', 'Speed', 'RPM', 'FuelLevel'],
            'time_model': ['ShiftDate', 'TimeStamp', 'Status'],
            'cycle': []  # Permitir dataset vacío
        }

    def _detect_truck_id(self) -> str:
        for df in self.datasets.values():
            if not df.empty and 'Equipment' in df.columns:
                return df['Equipment'].iloc[0]
        return 'N/A'

    def generate_comparison_table(self) -> Dict[str, Dict[str, str]]:
        results = {}
        for dtype in ['sensor', 'time_model', 'cycle']:
            df = self.datasets.get(dtype, pd.DataFrame())
            
            rango_temporal = 'N/A'
            if not df.empty and 'ShiftDate' in df.columns:
                valid_dates = df['ShiftDate'].dropna()
                if not valid_dates.empty:
                    rango_temporal = f"{valid_dates.min().date()} a {valid_dates.max().date()}"

            results[dtype] = {
                'Dimensiones': f"{len(df):,} x {len(df.columns)}" if not df.empty else 'N/A',
                'Valores Nulos': f"{df.isnull().sum().sum():,}" if not df.empty else 'N/A',
                'Duplicados': f"{df.duplicated().sum():,}" if not df.empty else 'N/A',
                'Rango Temporal': rango_temporal
            }
        return results

class ETLDataProcessor:
    def __init__(self, raw_datasets: Dict[str, pd.DataFrame]):
        self.raw_validator = DataValidator(raw_datasets)
        self.processed_datasets = {}
        self.processed_validator = None

    def _clean_data(self) -> None:
        self.processed_datasets = {k: v.copy() for k, v in self.raw_validator.datasets.items()}
        
        for dtype, df in self.processed_datasets.items():
            if df.empty:
                continue

            # Mantener registros con al menos una fecha válida
            if 'ShiftDate' in df.columns and 'TimeStamp' in df.columns:
                df = df[df[['ShiftDate', 'TimeStamp']].notna().any(axis=1)]

            # Limpieza específica
            if dtype == 'sensor':
                df = df.dropna(subset=['Speed', 'RPM', 'FuelLevel'], how='all')
            elif dtype == 'time_model':
                df = df.dropna(subset=['Status'])

            self.processed_datasets[dtype] = df

    def _transform_columns(self) -> None:
        for dtype, df in self.processed_datasets.items():
            if df.empty:
                continue

            # Conversión de tipos numéricos
            if dtype == 'sensor':
                df['Speed'] = pd.to_numeric(df['Speed'], errors='coerce')
                df['RPM'] = pd.to_numeric(df['RPM'], errors='coerce')

            # Componentes temporales
            if 'ShiftDate' in df.columns:
                df['ShiftYear'] = df['ShiftDate'].dt.year
                df['ShiftMonth'] = df['ShiftDate'].dt.month
                df['ShiftDay'] = df['ShiftDate'].dt.day
            if 'TimeStamp' in df.columns:
                df['Hour'] = df['TimeStamp'].dt.hour

            self.processed_datasets[dtype] = df

    def _apply_schema(self) -> None:
        COLUMN_MAPPING = {
            'sensor': [
                'ShiftYear', 'ShiftMonth', 'ShiftDay', 'Hour',
                'Speed', 'RPM', 'FuelLevel', 'FuelLevelLiters'
            ],
            'time_model': [
                'ShiftYear', 'ShiftMonth', 'ShiftDay', 'Hour',
                'Status', 'Category', 'Event'
            ],
            'cycle': []  # Sin columnas requeridas
        }

        for dtype, df in self.processed_datasets.items():
            if df.empty:
                continue

            allowed_cols = [col for col in COLUMN_MAPPING[dtype] if col in df.columns]
            self.processed_datasets[dtype] = df[allowed_cols] if allowed_cols else df

    def run_etl(self) -> 'ETLDataProcessor':
        self._clean_data()
        self._transform_columns()
        self._apply_schema()
        
        self.processed_validator = DataValidator(self.processed_datasets)
        schema_issues = self.processed_validator.validate_schema()
        
        if any(schema_issues.values()):
            print("⚠️ ALERTA: Problemas de esquema detectados")
            for dtype, issues in schema_issues.items():
                print(f"- {dtype.upper()}: {', '.join(issues)}")
        
        return self

    def show_comparison(self) -> None:
        raw_table = self.raw_validator.generate_comparison_table()
        processed_table = self.processed_validator.generate_comparison_table()
        
        print(f"\n{'='*60}")
        print("🚀 COMPARATIVO ETL COMPLETO".center(60))
        print(f"{'='*60}\n")
        
        for dtype in ['sensor', 'time_model', 'cycle']:
            print(f"📊 {dtype.upper()} ".ljust(60, '▬'))
            print(f"{'MÉTRICA':<25} | {'ORIGINAL':<15} | {'PROCESADO':<15}")
            print(f"{'-'*60}")
            
            for metric in ['Dimensiones', 'Valores Nulos', 'Duplicados', 'Rango Temporal']:
                orig = raw_table[dtype].get(metric, 'N/A')
                proc = processed_table[dtype].get(metric, 'N/A')
                print(f"{metric:<25} | {orig:<15} | {proc:<15}")
            print("\n")




# Flujo de uso
loader = DataLoader(DATA_DIR, "T-210")
raw_datasets = loader.load_data()

# Inspeccionar cada dataset
for dtype, df in raw_datasets.items():
    print(f"\n{'='*50}")
    print(f"DATASET: {dtype.upper()}".center(50))
    print(f"{'='*50}\n")
    
    if df.empty:
        print("¡Dataset vacío!")
        continue
    
    # Mostrar tipos de datos
    print(" TIPOS DE DATOS ".center(50, "-"))
    print(df.dtypes)
    
    # Mostrar primeras filas
    print("\n PRIMERAS 3 FILAS ".center(50, "-"))
    print(df.head(3))
    
    # Mostrar valores nulos por columna
    print("\n VALORES NULOS ".center(50, "-"))
    print(df.isnull().sum())
    
    # Mostrar información general (incluye memoria usada)
    print("\n INFORMACIÓN GENERAL ".center(50, "-"))
    df.info()



# ----------------------------------------------------------
# Bloque de Pruebas Mejorado con Diagnóstico Extendido
# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        # Configuración inicial
        import os
        import sys
        import traceback
        from time import perf_counter
        from IPython.display import display  # Solo para Jupyter
        
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        loader = ExploreDataLoader(project_root)
        
        # Configuración de prueba
        TEST_PARAMS = {
            'truck_name': 'T-210',
            'metrics': ['fuel', 'rpm'],
            'years': [ 2025]
        }

        # ----------------------------------------------------------
        # Ejecución y Monitoreo
        # ----------------------------------------------------------
        print(f"\n{'='*50}")
        print("🚚 INICIO DE PRUEBA - CARGA DE DATOS DE CAMIONES")
        print(f"{'='*50}\n")
        
        # 1. Descubrimiento inicial
        print("[Fase 1] Descubrimiento de recursos:")
        print(f"• Directorio base: {project_root}")
        print(f"• Camiones detectados: {loader.available_trucks}")
        
        # 2. Búsqueda de archivos
        patterns = loader._generate_file_patterns(**TEST_PARAMS)
        files = loader._find_matching_files(patterns)
        print(f"\n[Fase 2] Localización de archivos:")
        print(f"• Patrones usados: {len(patterns)}")
        print(f"• Archivos encontrados: {len(files)}")
        print("  Muestra de archivos:", [os.path.basename(f) for f in files[:3]] + (['...'] if len(files)>3 else []))
        
        # 3. Carga de datos
        print(f"\n[Fase 3] Carga y transformación:")
        start_time = perf_counter()
        df = loader.load_data(**TEST_PARAMS)
        load_time = perf_counter() - start_time
        
        # ----------------------------------------------------------
        # Análisis Post-Carga
        # ----------------------------------------------------------
        print(f"\n[Fase 4] Resultados y calidad de datos:")
        print(f"✅ Carga completada en {load_time:.2f} segundos")
        print(f"\n📦 Estructura del dataset:")
        print(f"• Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        print(f"• Rango temporal: {df['created_at_local'].min().strftime('%Y-%m-%d')} - {df['created_at_local'].max().strftime('%Y-%m-%d')}")
        print(f"• Métricas incluidas: {df['metric'].unique().tolist()}")
        
        print("\n🔍 Muestra estratificada (2 registros por métrica):")

        # Enfoque alternativo sin usar groupby.apply
        stratified_sample = pd.DataFrame()  # DataFrame vacío para almacenar resultados
        
        # Iterar sobre cada métrica única
        for m in df['metric'].unique():
            # Seleccionar 2 filas aleatorias para la métrica actual
            sample = df[df['metric'] == m].sample(n=2, random_state=1)
            # Concatenar con resultados anteriores
            stratified_sample = pd.concat([stratified_sample, sample])
        
        # Seleccionar y formatear columnas
        stratified_sample = stratified_sample[['created_at_local', 'metric', 'value', 'speed']].reset_index(drop=True)
        
        # Mostrar con formato
        display(stratified_sample.style
               .format({'value': '{:.2f}', 'speed': '{:.0f}'})
               .set_caption("Muestra estratificada por tipo de métrica")
               .hide(axis='index'))
                
        print("\n🧹 Calidad de datos:")
        print("• Valores nulos por columna:")
        null_counts = df.isna().sum()
        print(null_counts[null_counts > 0].to_string() or "  - Sin valores nulos")
        print(f"• Registros duplicados: {df.duplicated().sum()}")
        
        print("\n📊 Distribución clave:")
        print("• Velocidad (resumen estadístico):")
        print(df['speed'].describe().to_string())
        print("\n• Frecuencia de métricas:")
        print(df['metric'].value_counts().to_string())
        
        print(f"\n{'='*50}")
        print("✅ PRUEBA COMPLETADA SATISFACTORIAMENTE")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n{'='*50}")
        print(f"❌ ERROR EN PRUEBA: {str(e)}")
        traceback.print_exc()
        sys.exit(1)