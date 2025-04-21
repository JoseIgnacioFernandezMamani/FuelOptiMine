import pandas as pd
import os
from datetime import datetime

class FuelDataETL:
    def __init__(self):
        self.DATA_DIR = os.path.abspath(os.path.join("..", "data-set", "val_data_sensor", "raw_data"))
        # Updated to match actual format in the data
        self.VALID_DESCRIPTIONS = {'CAT - 789C', 'CAT - 793D'}
        self.TRUCK_PREFIXES = {str(i).zfill(3) for i in range(210, 226)} | {str(i).zfill(3) for i in range(230, 244)}
        self.raw_df = pd.DataFrame()
        self.transformed_df = pd.DataFrame()
        
    def extract(self):
        """Cargar todos los archivos de despacho relevantes"""
        file_list = [
            f for f in os.listdir(self.DATA_DIR)
            if f.startswith('despacho') and f.endswith(('.xls', '.xlsx'))
        ]
        dfs = []
        for file in file_list:
            try:
                file_path = os.path.join(self.DATA_DIR, file)
                engine = 'openpyxl' if file.endswith('.xlsx') else None
                
                df = pd.read_excel(
                    file_path,
                    engine=engine,
                    usecols=['Veh', 'Descripcion', 'fin_desp', 'volumCorregido']
                )
                # Add the source file information to the dataframe
                df['source_file'] = file
                dfs.append(df)
                print(f"✅ {file} cargado correctamente")
            except Exception as e:
                print(f"❌ Error cargando {file}: {str(e)}")
                continue
        if dfs:
            self.raw_df = pd.concat(dfs, ignore_index=True)
            print(f"\n📦 Datos brutos combinados: {len(self.raw_df)} registros")
        else:
            raise ValueError("No se encontraron archivos válidos para procesar")
    
    def transform(self):
        """Aplicar todas las transformaciones necesarias"""
        if self.raw_df.empty:
            raise ValueError("No hay datos para transformar")
        
        # Limpieza inicial
        df = self.raw_df.copy()
        df = df.dropna(subset=['Veh', 'Descripcion', 'fin_desp'])
        
        # Filtrado por descripción
        df['Descripcion'] = df['Descripcion'].str.upper().str.strip()
        valid_desc = {d.upper() for d in self.VALID_DESCRIPTIONS}
        df = df[df['Descripcion'].isin(valid_desc)]
        
        # Procesamiento de Vehículos - Manejo flexible del formato
        # Extraemos hasta 3 dígitos consecutivos donde sea que estén en el campo Veh
        df['Veh_Prefix'] = df['Veh'].astype(str).str.extract(r'(\d{3})')[0]
        df = df[df['Veh_Prefix'].isin(self.TRUCK_PREFIXES)]
        df['Veh'] = 'T-' + df['Veh_Prefix']
        df = df.drop('Veh_Prefix', axis=1)
        
        # Transformación de fechas - No convertir si ya son datetime
        if not pd.api.types.is_datetime64_any_dtype(df['fin_desp']):
            df['fin_desp'] = pd.to_datetime(
                df['fin_desp'],
                errors='coerce',
                dayfirst=True
            )
        
        df = df.dropna(subset=['fin_desp'])
        
        # Crear nuevas columnas temporales
        df = df.assign(
            ShiftDate=df['fin_desp'].dt.normalize(),
            TimeStamp=df['fin_desp'].dt.time
        )
        
        # Extraer el origen (parte después del underscore y antes de la extensión)
        df['Origin'] = df['source_file'].apply(
            lambda filename: filename.split('_', 1)[1].split('.')[0] if '_' in filename else 'UNKNOWN'
        )
        
        # Renombrar y seleccionar columnas finales
        self.transformed_df = df.rename(columns={
            'Veh': 'Equipment',
            'volumCorregido': 'FuelLevelLiters'
        })[["ShiftDate", "TimeStamp", "Equipment", "FuelLevelLiters", "Origin"]]
        
        # Ordenar por TimeStamp
        self.transformed_df = self.transformed_df.sort_values(by=['ShiftDate', 'TimeStamp'])
        
        # Validación final
        self._validate_data()
        print(f"\n✨ Transformación completada: {len(self.transformed_df)} registros válidos")

    
    def _validate_data(self):
        """Validación de consistencia de datos"""
        # Verificar rangos de fechas
        min_date = self.transformed_df['ShiftDate'].min() if not self.transformed_df.empty else None
        max_date = self.transformed_df['ShiftDate'].max() if not self.transformed_df.empty else None
        print(f"📅 Rango de fechas: {min_date} - {max_date}")
        
        # Verificar valores únicos de equipos
        unique_equipment = self.transformed_df['Equipment'].unique() if not self.transformed_df.empty else []
        equipment_str = ', '.join(sorted(unique_equipment)) if len(unique_equipment) > 0 else "Ninguno"
        print(f"🚚 Equipos detectados: {equipment_str}")
        
        # Verificar valores de combustible
        fuel_stats = self.transformed_df['FuelLevelLiters'].describe() if not self.transformed_df.empty else None
        print(f"⛽ Estadísticas de combustible:\n{fuel_stats}")

    def load(self, output_path='fuel_data.csv'):
        
        if not self.transformed_df.empty:
            self.transformed_df.to_csv(output_path, index=False)
            print(f"\n💾 Datos guardados en: {output_path}")
            return True
        print("⚠️ No hay datos para guardar")
        return False

    
if __name__ == '__main__':
    # Ejecución del pipeline completo
    etl = FuelDataETL()
    
    etl.extract()
    etl.transform()
    etl.load()


import pandas as pd
import os
import re
from datetime import datetime

class FuelDataETL:
    def __init__(self):
        self.DATA_DIR = os.path.abspath(os.path.join("..", "data-set", "val_data_sensor", "raw_data"))
        self.valid_equipment = [f"T-{str(i).zfill(3)}" for i in list(range(210, 226)) + list(range(230, 244))]
        self.final_columns = ['ShiftDate', 'TimeStamp', 'Equipment', 'FuelLevelLiters']
        self.transformed_df = pd.DataFrame(columns=self.final_columns)

    def extract_transform(self):
        """Procesa todos los archivos equipo_*.xlsx"""
        equipment_files = [f for f in os.listdir(self.DATA_DIR) if f.startswith('equipo') and f.endswith('.xlsx')]
        
        for file in equipment_files:
            file_path = os.path.join(self.DATA_DIR, file)
            xls = pd.ExcelFile(file_path)
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                self._process_sheet(df, file, sheet_name)
        
        return self.transformed_df

    def _process_sheet(self, df, filename, sheetname):
        """Procesa cada hoja del archivo Excel"""
        date_pattern = re.compile(r'EQUIPO\s+\w+\s+(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE)
        
        for row_idx in range(3):  # Buscar en primeras 3 filas
            for col_idx in range(df.shape[1]):
                cell_value = str(df.iloc[row_idx, col_idx])
                date_match = date_pattern.search(cell_value)
                
                if date_match:
                    shift_date = self._parse_date(date_match.group(1))
                    self._process_columns(df, shift_date, col_idx, row_idx)
                    break

    def _parse_date(self, date_str):
        """Convierte la fecha al formato correcto"""
        try:
            return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
        except:
            return None

    def _process_columns(self, df, shift_date, start_col, header_row):
        """Procesa las columnas de datos"""
        if not shift_date:
            return
            
        # Buscar columna CODIGO EQ.
        for col in range(start_col, min(start_col+6, df.shape[1])):
            if "CODIGO EQ." in str(df.iloc[header_row+1, col]):
                code_col = col
                fuel_col = code_col + 2  # Saltar RECEPCION
                break
        else:
            return

        # Procesar filas de datos
        for row in range(header_row+2, df.shape[0]):
            equipment_code = str(df.iloc[row, code_col]).strip()
            
            if not equipment_code.isdigit():
                continue
                
            equipment = f"T-{equipment_code.zfill(3)}"
            
            if equipment in self.valid_equipment:
                fuel = df.iloc[row, fuel_col]
                
                if pd.notna(fuel) and str(fuel).replace('.','',1).isdigit():
                    new_row = {
                        'ShiftDate': shift_date,
                        'TimeStamp': '00:00:00',
                        'Equipment': equipment,
                        'FuelLevelLiters': float(fuel)
                    }
                    
                    self.transformed_df = pd.concat(
                        [self.transformed_df, pd.DataFrame([new_row])],
                        ignore_index=True
                    )

    def load(self, output_file='fuel_data.csv'):
        """Guarda los datos en CSV"""
        if not self.transformed_df.empty:
            self.transformed_df.to_csv(output_file, index=False)
            print(f"Datos guardados en {output_file}")
        else:
            print("No hay datos para guardar")

# Ejecución del proceso
if __name__ == "__main__":
    etl = FuelDataETL()
    etl.extract_transform()
    
    print("\nMuestra de datos transformados:")
    print(etl.transformed_df.head())
    
    etl.load()











import os
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import logging
import sys
from tqdm import tqdm  # Para mostrar barras de progreso

# Configurar logging para mostrar en archivo y consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler(sys.stdout)  # Esto lo muestra en consola
    ]
)
logger = logging.getLogger()

# Parámetros de conexión
server = 'mscexagonrep'
database = 'Msc_Dev'
driver = 'ODBC Driver 17 for SQL Server'

# Cadena de conexión
connection_string = f'mssql+pyodbc://@{server}/{database}?driver={driver}&TrustServerCertificate=yes&trusted_connection=yes'
engine = create_engine(connection_string)

# Lista de camiones (ID: nombre)
camiones = {
    287: "T-211", 288: "T-212", 289: "T-213", 290: "T-214", 291: "T-215", 
    292: "T-216", 293: "T-217", 294: "T-218", 295: "T-219", 296: "T-220", 
    297: "T-221", 298: "T-222", 299: "T-223", 300: "T-224", 301: "T-225", 
    302: "T-230", 303: "T-231", 304: "T-232", 305: "T-233", 312: "T-235", 
    313: "T-236", 314: "T-237", 315: "T-238", 316: "T-239", 317: "T-240", 
    318: "T-241", 319: "T-242", 320: "T-243"
}

# Tipos de datos a recopilar (SP disponibles)
tipos_datos = ["sensor", "time_model"]

# Procedimientos almacenados correspondientes a cada tipo
stored_procedures = {
    "sensor": "[Model].[GetEquipmentSensorData]",
    "time_model": "[Model].[TimeModel]"
}

# Fecha de inicio y fin para la recopilación de datos
fecha_inicio = datetime(2024, 2, 1)
fecha_fin = datetime(2025, 3, 3)

# Contador para seguimiento
total_archivos = 0
archivos_exitosos = 0
archivos_sin_datos = 0

def get_data_for_period(truck_id, data_type, start_date, end_date):
    """Obtiene datos para un camión y tipo específico en un período dado"""
    
    # Formatear fechas para la consulta
    fecha_ini = start_date.strftime('%Y-%m-%d 07:00:00')
    fecha_fin = end_date.strftime('%Y-%m-%d 06:59:59')
    
    # Consulta SQL para ejecutar el SP correspondiente
    sp = stored_procedures[data_type]
    query = f"""
    EXEC {sp} 
        @FechaIni = '{fecha_ini}',
        @FechaFin = '{fecha_fin}',
        @EquipmentID = {truck_id}
    """
    
    try:
        # Ejecutar consulta
        df = pd.read_sql(query, engine)
        
        # Verificar si hay datos
        if df.empty:
            print(f"✘ No se encontraron datos para {camiones[truck_id]} tipo {data_type} del {start_date.strftime('%Y-%m')}.")
            logger.info(f"No se encontraron datos para {camiones[truck_id]} tipo {data_type} del {start_date.strftime('%Y-%m')}.")
            return None
        
        logger.info(f"Datos obtenidos para {camiones[truck_id]} tipo {data_type} del {start_date.strftime('%Y-%m')}: {len(df)} registros.")
        return df
    
    except SQLAlchemyError as e:
        print(f"✘ ERROR al obtener datos para {camiones[truck_id]} tipo {data_type}: {e}")
        logger.error(f"Error al obtener datos para {camiones[truck_id]} tipo {data_type}: {e}")
        return None

def save_data_to_csv(df, truck_name, data_type, period):
    """Guarda los datos en un archivo CSV con el formato especificado"""
    
    # Determinar el año para ubicar en la carpeta correcta
    year = period[:4]
    
    # Crear estructura de directorios si no existe
    base_dir = f"data-set/train_data_{data_type}/test_{year}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Nombre del archivo según el formato: camion_tipo_año-mes.csv
    filename = f"{truck_name}_{data_type}_{period}.csv"
    file_path = os.path.join(base_dir, filename)
    
    # Guardar a CSV
    df.to_csv(file_path, index=False)
    print(f"✓ Guardado: {file_path} ({len(df)} registros)")
    logger.info(f"Datos guardados en {file_path}")
    
    return file_path

def print_progress_header():
    """Imprime un encabezado de progreso general"""
    print("\n" + "="*70)
    print(f"PROGRESO GENERAL: {archivos_exitosos}/{total_archivos} archivos procesados correctamente")
    print(f"                  {archivos_sin_datos} consultas sin datos")
    print("="*70 + "\n")

def main():
    """Función principal para recopilar y guardar todos los datos"""
    global total_archivos, archivos_exitosos, archivos_sin_datos
    
    # Calcular el número total de operaciones a realizar
    total_meses = ((fecha_fin.year - fecha_inicio.year) * 12 + 
                   fecha_fin.month - fecha_inicio.month)
    total_operaciones = total_meses * len(camiones) * len(tipos_datos)
    total_archivos = total_operaciones
    
    print(f"\n🚀 Comenzando proceso de recolección de datos para {len(camiones)} camiones")
    print(f"📅 Período: {fecha_inicio.strftime('%Y-%m')} a {fecha_fin.strftime('%Y-%m')}")
    print(f"🔄 Total de operaciones a realizar: {total_operaciones}\n")
    
    # Iterar por cada mes desde la fecha de inicio hasta la fecha de fin
    current_date = fecha_inicio
    mes_count = 1
    
    while current_date < fecha_fin:
        # Calcular el primer día del siguiente mes
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)
        
        # Período para el nombre del archivo
        period = current_date.strftime('%Y-%m')
        
        print(f"\n📋 Procesando período {mes_count}/{total_meses}: {period}")
        logger.info(f"Procesando período: {period}")
        
        # Para cada camión, obtener y guardar datos de cada tipo
        for truck_id, truck_name in camiones.items():
            for data_type in tipos_datos:
                # Obtener datos
                df = get_data_for_period(truck_id, data_type, current_date, next_month)
                
                # Si hay datos, guardarlos
                if df is not None and not df.empty:
                    save_data_to_csv(df, truck_name, data_type, period)
                    archivos_exitosos += 1
                else:
                    archivos_sin_datos += 1
                
                # Mostrar progreso cada 10 operaciones
                if (archivos_exitosos + archivos_sin_datos) % 10 == 0:
                    print_progress_header()
        
        # Avanzar al siguiente mes
        current_date = next_month
        mes_count += 1
    
    # Mostrar resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN FINAL DEL PROCESO")
    print("="*70)
    print(f"✓ Total de archivos guardados: {archivos_exitosos}")
    print(f"✘ Total de consultas sin datos: {archivos_sin_datos}")
    print(f"🏁 Proceso completado! Los datos se almacenaron en la carpeta 'data-set'")
    print("="*70 + "\n")
    
    logger.info("Proceso de recolección de datos completado.")

if __name__ == "__main__":
    try:
        logger.info("Iniciando proceso de recolección de datos...")
        main()
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        logger.error(f"Error en el proceso principal: {e}")
    finally:
        logger.info("Proceso finalizado.")