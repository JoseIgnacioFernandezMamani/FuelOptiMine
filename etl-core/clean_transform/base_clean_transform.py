class ETLDataProcessor:
    def __init__(self, truck: str):
        self.processed = {}
        self.column_mapping = COLUMN_MAPPING  
        self.raw_datasets = DataLoader(truck).load_data()
        
        # 2. Validación de entrada
        if not isinstance(self.raw_datasets, dict):
            raise TypeError("Se esperaba un diccionario de DataFrames")
            
    def _clean_data(self):
        """Limpieza robusta respetando las reglas especificadas"""
        for dtype, df in self.raw_datasets.items():
            if df.empty:
                self.processed[dtype] = df
                continue

            # 1. Validar columnas obligatorias
            required_cols = ['ShiftDate', 'TimeStamp']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Columnas críticas faltantes en {dtype}: {missing}")

            # 2. Eliminar filas con ShiftDate o TimeStamp nulos (ambos deben tener valor)
            df = df.dropna(subset=required_cols, how='any')

            # 3. Limpieza específica por tipo de dataset
            if dtype == 'sensor':
                # Eliminar filas donde todos los campos numéricos son nulos
                df = df.dropna(subset=['Speed', 'RPM', 'FuelLevel'], how='all')
                
            elif dtype == 'time_model':
                # Eliminar filas sin Status
                df = df.dropna(subset=['Status'])

            self.processed[dtype] = df
        
    def _transform_columns(self):
        """Transformaciones de formato y tipos"""
        for dtype, df in self.processed.items():
            if df.empty:
                continue
                
            # Convertir fechas a formatos correctos
            df['ShiftDate'] = pd.to_datetime(df['ShiftDate'])
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp']).dt.tz_localize(None)
        # 2. Crear datetime completo (ShiftDate + TimeStamp)
            df['FullDateTime'] = df.apply(
                lambda row: row['ShiftDate'].replace(
                    hour=row['TimeStamp'].hour,
                    minute=row['TimeStamp'].minute,
                    second=row['TimeStamp'].second
                ), axis=1
            )
            # columnas agregadas
            df.insert(loc=1, column="ShiftYear", value=df['ShiftDate'].dt.year)
            df.insert(loc=2, column="ShiftMonth", value=df['ShiftDate'].dt.month)
            df.insert(loc=3, column="ShiftDay", value=df['ShiftDate'].dt.day)

            # modificar la columna a solo hora minuto y segundo
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp']).dt.time
            
            # Procesar SOLO las celdas nulas en cada columna
            for column in df.columns:
                # Crear una máscara de valores nulos para la columna
                mascara_nulos = df[column].isnull()
                
                # Solo si hay al menos una celda nula procedemos
                if mascara_nulos.any():
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # Reemplazar SOLO las celdas nulas con 0
                        df.loc[mascara_nulos, column] = 0
                    elif column == 'TimeStamp' and mascara_nulos.any():
                        # Para TimeStamp nulos, usar medianoche
                        from datetime import time
                        df.loc[mascara_nulos, column] = time(0, 0, 0)
                    elif pd.api.types.is_string_dtype(df[column]) or df[column].dtype == 'object':
                        # Para celdas nulas en columnas de texto/objeto
                        df.loc[mascara_nulos, column] = "N/A"
                
            # agregar metadatos
            df.insert(loc=len(df.columns), column="data_type", value=dtype)

            self.processed[dtype] = df
                
    def run_etl(self) -> dict:
        self._clean_data()
        self._transform_columns()
        return self.processed



etl_processor = ETLDataProcessor("T-234")
processed_data = etl_processor.run_etl()
sensor_df = processed_data['sensor']
sensor_df.head(5)















# =============================================
# Temporal Transformation & Cleaning (Versión Corregida)
# =============================================

def clean_and_analyze_temporal_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y análisis avanzado de datos temporales."""
    
    # --------------------------------------------------
    # 4.1 Validación inicial
    # --------------------------------------------------
    required_columns = ['created_at_local', 'metric', 'value']
    if not all(col in raw_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in raw_df.columns]
        raise KeyError(f"Columnas requeridas faltantes: {missing}")

    # --------------------------------------------------
    # 4.2 Conversión y normalización temporal
    # --------------------------------------------------
    # Convertir a datetime con manejo de errores
    raw_df['created_at_local'] = pd.to_datetime(
        raw_df['created_at_local'],
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce',
        utc=True
    ).dt.tz_convert('America/Santiago')
    
    # --------------------------------------------------
    # 4.3 Análisis de calidad de datos temporales
    # --------------------------------------------------
    # Identificar registros problemáticos
    temporal_issues = raw_df[raw_df['created_at_local'].isna()]
    
    print("\n🔔 Alertas de Calidad Temporal:")
    if not temporal_issues.empty:
        print(f"Registros temporales inválidos: {len(temporal_issues):,}")
        print("Distribución por métrica:")
        print(temporal_issues['metric'].value_counts(dropna=False).to_string())
        
        # Guardar diagnóstico
        os.makedirs('diagnosticos', exist_ok=True)
        temporal_issues.to_csv('diagnosticos/registros_temporales_problematicos.csv', index=False)
    else:
        print("✅ Todos los registros tienen marcas temporales válidas")

    # --------------------------------------------------
    # 4.4 Limpieza de datos
    # --------------------------------------------------
    clean_df = raw_df.dropna(subset=['created_at_local']).copy()
    
    # Validación básica post-limpieza
    print("\n✅ Post-limpieza:")
    print(f"- Registros originales: {len(raw_df):,}")
    print(f"- Registros válidos: {len(clean_df):,} ({len(clean_df)/len(raw_df)*100:.1f}%)")
    
    # --------------------------------------------------
    # 4.5 Análisis de distribución temporal
    # --------------------------------------------------
    print("\n📊 Características Temporales:")
    
    # Estadísticas por métrica
    metric_stats = clean_df.groupby('metric')['created_at_local'].agg(
        primera_medicion='min',
        ultima_medicion='max',
        total_registros='count',
        frecuencia_media=lambda x: x.diff().median().total_seconds() if len(x) > 1 else 0
    ).reset_index()
    
    print("\nEstadísticas por Métrica:")
    print(metric_stats.to_string(index=False))
    
    # Análisis de densidad diaria
    print("\n📈 Densidad de Registros por Día:")
    clean_df['fecha'] = clean_df['created_at_local'].dt.date
    daily_density = clean_df.groupby(['fecha', 'metric']).size().unstack(fill_value=0)
    print(daily_density.describe().to_string())
    
    return clean_df

# Uso en el flujo principal
if __name__ == "__main__":
    try:
        # Cargar datos
        loader = ExploreDataLoader(os.path.join('..'))
        raw_data = loader.load_data(
            truck_name='T-210',
            metrics=['rpm'],
            years=[2024, 2025]
        )
        
        # Procesamiento temporal
        clean_data = clean_and_analyze_temporal_data(raw_data)
        
    except Exception as e:
        print(f"Error en proceso: {str(e)}")
        sys.exit(1)