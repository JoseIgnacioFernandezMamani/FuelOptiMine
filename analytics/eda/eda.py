import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, widgets
from datetime import time, timedelta

class TruckEDA:
    """
    Clase para realizar análisis exploratorio de datos para camiones.
    Integra análisis estático e interactivo para múltiples tipos de datos.
    """
    def __init__(self, data_dict: dict, truck_id: str):
        """
        Inicializa el análisis con los datasets procesados.
        
        Args:
            data_dict: Diccionario con los datasets procesados ('sensor', 'time_model', 'cycle')
            truck_id: Identificador del camión a analizar
        """
        self.data = data_dict
        self.truck_id = truck_id
        # Verificar existencia de datos
        for key, df in self.data.items():
            if df.empty:
                print(f"⚠️ El dataset '{key}' está vacío")
                
        # Crear directorio para guardar gráficos
        self.output_dir = f"reports/truck_eda/{truck_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_full_eda(self):
        """Ejecuta el análisis exploratorio completo"""
        print(f"\n{'='*50}")
        print(f"🔍 ANÁLISIS EXPLORATORIO PARA CAMIÓN {self.truck_id}")
        print(f"{'='*50}")
        
        # 1. Resumen básico de los datos
        self._print_summary()
        
        # 2. Análisis de integridad y calidad
        self._analyze_data_quality()
        
        # 3. Análisis temporal
        self._temporal_analysis()
        
        # 4. Análisis de rendimiento operativo
        self._performance_analysis()
        
        # 5. Análisis espacial (si hay coordenadas)
        self._spatial_analysis()
        
        # 6. Correlaciones entre variables clave
        self._correlation_analysis()
        
        print(f"\n✅ Análisis EDA completado. Resultados guardados en: {self.output_dir}")
    
    def _print_summary(self):
        """Muestra un resumen general de los datasets"""
        print("\n📊 RESUMEN DE DATASETS")
        print("-" * 40)
        
        for key, df in self.data.items():
            if df.empty:
                continue
                
            print(f"\n• Dataset: {key.upper()}")
            print(f"  - Registros: {len(df):,}")
            print(f"  - Variables: {len(df.columns)}")
            
            # Rango temporal si existe
            if 'ShiftDate' in df.columns:
                min_date = df['ShiftDate'].min()
                max_date = df['ShiftDate'].max()
                date_range = (max_date - min_date).days + 1
                print(f"  - Cobertura temporal: {min_date.date()} a {max_date.date()} ({date_range} días)")
            
            # Tipos de variables
            dtypes = df.dtypes.value_counts()
            print(f"  - Tipos de datos: {dict(dtypes)}")
    
    def _analyze_data_quality(self):
        """Analiza la calidad e integridad de los datos"""
        print("\n🧐 ANÁLISIS DE CALIDAD DE DATOS")
        print("-" * 40)
        
        for key, df in self.data.items():
            if df.empty:
                continue
                
            print(f"\n• Dataset: {key.upper()}")
            
            # 1. Valores duplicadoos
            duplicates = df.duplicated().sum()
            print(f"  - Registros duplicados: {duplicates} ({duplicates/len(df)*100:.2f}%)")
            
            # 2. Completitud
            completeness = (df.count() / len(df) * 100).round(2)
            low_completeness = completeness[completeness < 98].to_dict()
            if low_completeness:
                print(f"  - Variables con baja completitud: {low_completeness}")
            else:
                print(f"  - Completitud de variables: Todas >98% completas")
            
            # 3. Visualización de completitud
            if len(df.columns) > 5:  # Solo para datasets con muchas columnas
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
                plt.title(f'Patrón de valores faltantes - {key.upper()}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/missing_pattern_{key}.png", dpi=300)
                plt.close()
    
    def _temporal_analysis(self):
        """Realiza análisis temporal de los datos"""
        print("\n⏱️ ANÁLISIS TEMPORAL")
        print("-" * 40)
        
        # Análisis específico para datos de sensores
        sensor_df = self.data.get('sensor')
        if sensor_df is not None and not sensor_df.empty and 'TimeStamp' in sensor_df.columns:
            # Preparar datos diarios para histograma
            daily_data = self._prepare_daily_data(sensor_df)
            
            # Gráfico de actividad diaria
            fig = px.bar(
                daily_data, 
                x='dia_mes', 
                y='conteo', 
                color='mes',
                facet_row='año',
                title=f'Registros diarios del camión {self.truck_id}',
                labels={'dia_mes': 'Día del mes', 'conteo': 'Número de registros', 'mes': 'Mes'},
                template='plotly_white',
                height=600
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=5),
                hovermode='x unified'
            )
            
            # Guardar como HTML interactivo
            fig.write_html(f"{self.output_dir}/daily_activity.html")
            
            # Distribución horaria para análisis de patrones
            if 'ShiftDate' in sensor_df.columns and isinstance(sensor_df['TimeStamp'].iloc[0], time):
                sensor_df['hour'] = [t.hour if isinstance(t, time) else 0 for t in sensor_df['TimeStamp']]
                
                plt.figure(figsize=(12, 6))
                sns.countplot(x='hour', data=sensor_df, palette='viridis')
                plt.title(f'Distribución horaria de actividad - Camión {self.truck_id}')
                plt.xlabel('Hora del día')
                plt.ylabel('Cantidad de registros')
                plt.xticks(range(0, 24))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/hourly_distribution.png", dpi=300)
                plt.close()
                
                print(f"  • Análisis temporal completado. Se generaron visualizaciones de patrones diarios y horarios.")
    
    def _prepare_daily_data(self, df):
        """Prepara datos para análisis temporal diario"""
        # Asumimos que TimeStamp ya está procesado adecuadamente
        temp_df = df.copy()
        
        # Crear columna de fecha a partir de ShiftDate si existe
        if 'ShiftDate' in temp_df.columns:
            # Extraer componentes
            temp_df['mes'] = temp_df['ShiftDate'].dt.month_name(locale='es')
            temp_df['dia_mes'] = temp_df['ShiftDate'].dt.day
            temp_df['año'] = temp_df['ShiftDate'].dt.year
            
            # Agrupar por día
            daily_counts = temp_df.groupby(['año', 'mes', 'dia_mes']).size().reset_index(name='conteo')
            
            # Orden de meses en español
            month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
            daily_counts['mes'] = pd.Categorical(daily_counts['mes'], categories=month_order, ordered=True)
            
            return daily_counts.sort_values(['año', 'mes', 'dia_mes'])
        else:
            return pd.DataFrame()  # Devolver DataFrame vacío si no hay fecha
    
    def _performance_analysis(self):
        """Analiza variables de rendimiento operativo"""
        print("\n🚚 ANÁLISIS DE RENDIMIENTO")
        print("-" * 40)
        
        sensor_df = self.data.get('sensor')
        if sensor_df is not None and not sensor_df.empty:
            perf_metrics = ['Speed', 'RPM', 'FuelLevel']
            available_metrics = [col for col in perf_metrics if col in sensor_df.columns]
            
            if available_metrics:
                print(f"  • Analizando métricas: {', '.join(available_metrics)}")
                
                # Crear subplots para distribuciones
                fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4*len(available_metrics)))
                if len(available_metrics) == 1:
                    axes = [axes]  # Convertir a lista si solo hay un eje
                
                for i, metric in enumerate(available_metrics):
                    # Histograma con KDE
                    sns.histplot(sensor_df[metric].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribución de {metric}')
                    axes[i].grid(linestyle='--', alpha=0.7)
                    
                    # Añadir estadísticas
                    stats = sensor_df[metric].describe()
                    stats_text = (f"Media: {stats['mean']:.2f}, Mediana: {stats['50%']:.2f}\n"
                                 f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                    axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, 
                               fontsize=9, va='top', ha='right', 
                               bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/performance_distributions.png", dpi=300)
                plt.close()
                
                # Gráfico interactivo de dispersión RPM vs Speed (si disponibles)
                if 'RPM' in sensor_df.columns and 'Speed' in sensor_df.columns:
                    fig = px.scatter(
                        sensor_df.sample(min(10000, len(sensor_df))),  # Muestreo para rendimiento
                        x='RPM', 
                        y='Speed',
                        color='FuelLevel' if 'FuelLevel' in sensor_df.columns else None,
                        opacity=0.6,
                        title=f'Relación RPM vs Velocidad - Camión {self.truck_id}',
                        template='plotly_white'
                    )
                    fig.write_html(f"{self.output_dir}/rpm_vs_speed.html")
    
    def _spatial_analysis(self):
        """Analiza datos espaciales si están disponibles"""
        print("\n🗺️ ANÁLISIS ESPACIAL")
        print("-" * 40)
        
        sensor_df = self.data.get('sensor')
        if sensor_df is not None and not sensor_df.empty:
            geo_cols = ['Latitude', 'Longitude']
            
            if all(col in sensor_df.columns for col in geo_cols):
                # Filtrar coordenadas válidas
                geo_df = sensor_df[
                    (sensor_df['Latitude'].between(-90, 90)) & 
                    (sensor_df['Longitude'].between(-180, 180))
                ].copy()
                
                if len(geo_df) > 0:
                    print(f"  • Registros con coordenadas válidas: {len(geo_df):,}")
                    
                    # Mapa interactivo de trayectorias
                    fig = px.scatter_mapbox(
                        geo_df.sample(min(5000, len(geo_df))),  # Muestreo para rendimiento
                        lat='Latitude',
                        lon='Longitude',
                        color='Speed' if 'Speed' in geo_df.columns else None,
                        size_max=10,
                        zoom=12,
                        title=f'Trayectorias del camión {self.truck_id}',
                        mapbox_style="carto-positron"
                    )
                    fig.write_html(f"{self.output_dir}/trajectories_map.html")
                else:
                    print("  • No se encontraron coordenadas geográficas válidas")
            else:
                print("  • No se encontraron columnas de coordenadas geográficas")
    
    def _correlation_analysis(self):
        """Analiza correlaciones entre variables operativas"""
        print("\n📊 ANÁLISIS DE CORRELACIONES")
        print("-" * 40)
        
        sensor_df = self.data.get('sensor')
        if sensor_df is not None and not sensor_df.empty:
            # Seleccionar columnas numéricas relevantes
            numeric_cols = sensor_df.select_dtypes(include=['number']).columns.tolist()
            
            # Filtrar columnas no relevantes para correlación
            exclude_cols = ['ShiftYear', 'ShiftMonth', 'ShiftDay']
            corr_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(corr_cols) >= 2:
                # Matriz de correlación
                corr_matrix = sensor_df[corr_cols].corr()
                
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    cmap='coolwarm',
                    annot=True, 
                    fmt=".2f",
                    center=0,
                    square=True, 
                    linewidths=.5
                )
                plt.title(f'Correlaciones entre variables operativas - Camión {self.truck_id}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300)
                plt.close()
                
                print(f"  • Matriz de correlación generada con {len(corr_cols)} variables")
            else:
                print("  • Insuficientes variables numéricas para análisis de correlación")
    
    def generate_interactive_dashboard(self):
        """Genera un dashboard interactivo para exploración temporal"""
        sensor_df = self.data.get('sensor')
        if sensor_df is None or sensor_df.empty:
            print("❌ No hay datos de sensores disponibles para el dashboard interactivo")
            return
        
        # Función para generar histograma diario interactivo
        def plot_daily_histogram_interactive(df):
            # Prepara datos
            hist_data = self._prepare_daily_data(df)
            if hist_data.empty:
                print("❌ No se pudieron preparar datos temporales")
                return
                
            meses_disponibles = hist_data['mes'].dropna().unique()
            
            @interact(
                Mes=widgets.Dropdown(
                    options=meses_disponibles,
                    description='Seleccionar Mes:',
                    style={'description_width': 'initial'}
                )
            )
            def plot_histogram(Mes):
                # Filtrar mes
                filtered = hist_data[hist_data['mes'] == Mes]
                fig = px.bar(
                    filtered,
                    x='dia_mes',
                    y='conteo',
                    color='año',
                    barmode='group',
                    title=f'Registros diarios de {self.truck_id} - {Mes}',
                    labels={'dia_mes': 'Día del Mes', 'conteo': 'Número de Registros'},
                    template='plotly_white'
                )
                fig.update_layout(
                    xaxis=dict(
                        tickmode='linear',
                        dtick=1,
                        range=[0.5, 31.5]
                    ),
                    hovermode='x unified',
                    height=500
                )
                # Línea de promedio
                avg = filtered['conteo'].mean()
                fig.add_hline(
                    y=avg,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f'Promedio: {avg:.1f}',
                    annotation_position="top right"
                )
                fig.show()
                
        return plot_daily_histogram_interactive(sensor_df)

# Función de ejecución principal
def run_truck_eda(processed_data: dict, truck_id: str, interactive: bool = False):
    """
    Ejecuta análisis exploratorio completo para un camión específico
    
    Args:
        processed_data: Diccionario con DataFrames procesados
        truck_id: ID del camión a analizar
        interactive: Si es True, genera visualizaciones interactivas adicionales
    """
    # Crear instancia de análisis
    eda = TruckEDA(processed_data, truck_id)
    
    # Ejecutar análisis completo
    eda.run_full_eda()
    
    # Generar dashboard interactivo si se solicita
    if interactive:
        print("\n🔍 Generando dashboard interactivo...")
        return eda.generate_interactive_dashboard()
    
    return None

# Ejemplo de uso:
# ---------------
# # 1. Cargar y procesar datos con ETLDataProcessor
etl = ETLDataProcessor(truck="T-210")
processed_data = etl.run_etl()
#
# # 2. Ejecutar análisis exploratorio
run_truck_eda(processed_data, "T-210", interactive=True)




























# =============================================
# Block 5 - Exploratory Data Analysis (EDA)
# =============================================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.api.types import is_numeric_dtype

class AdvancedEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def perform_full_analysis(self, temporal_col: str = 'created_at_local', 
                             value_col: str = 'value', save_path: str = 'reports/eda'):
        """Ejecuta análisis completo con visualizaciones interactivas y estáticas."""
        os.makedirs(save_path, exist_ok=True)
        
        print("\n🔎 Análisis Exploratorio Inicial (EDA)")
        self._show_basic_stats()
        self._temporal_analysis(temporal_col, save_path)
        self._distribution_analysis(value_col, save_path)
        self._correlation_analysis(save_path)
        self._outlier_detection(value_col, save_path)
        
    def _show_basic_stats(self):
        """Muestra estadísticas básicas del dataset."""
        # Importar pandas al principio del método para evitar el UnboundLocalError
        import pandas as pd
        
        print("\n📊 Estadísticas Básicas:")
        print(f"• Dimensiones: {self.df.shape[0]:,} registros x {self.df.shape[1]} variables")
        
        # Verificar tipo de dato temporal
        if not pd.api.types.is_datetime64_any_dtype(self.df['created_at_local']):
            raise TypeError("La columna temporal no está en formato datetime")
        
        print(f"• Rango Temporal: {self.df['created_at_local'].min()} a {self.df['created_at_local'].max()}")
        
        # Diagnóstico de versión
        print(f"Versión de pandas: {pd.__version__}")
        
        # Resumen estadístico
        print("\n📋 Resumen Estadístico:")
        
        # 1. Columnas numéricas
        print("\n[Variables Numéricas]")
        print(self.df.describe(include='number').to_string())
        
        # 2. Datetime específico usando método alternativo
        print("\n[Variable Temporal]")
        temporal_stats = pd.Series({
            'min': self.df['created_at_local'].min(),
            'max': self.df['created_at_local'].max(),
            'range': self.df['created_at_local'].max() - self.df['created_at_local'].min(),
            'count': self.df['created_at_local'].count(),
            'nunique': self.df['created_at_local'].nunique()
        })
        print(temporal_stats.to_string())
        
        # 3. Categóricas
        print("\n[Variables Categóricas]")
        print(self.df.describe(include=['object', 'category']).to_string())
        
    def _temporal_analysis(self, temporal_col: str, save_path: str):
        """Analiza patrones temporales con visualizaciones interactivas."""
        print("\n⏳ Análisis Temporal:")
        
        # Densidad temporal por métrica
        fig = px.density_heatmap(
            self.df,
            x=temporal_col,
            y='metric',
            title='Densidad de Registros por Hora y Métrica'
        )
        fig.write_html(f"{save_path}/temporal_density.html")
        
        # Tendencia temporal de valores
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=self.df,
            x=temporal_col,
            y='value',
            hue='metric',
            estimator='median',
            errorbar=None
        )
        plt.title('Tendencia Temporal de Valores (Mediana por Métrica)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/value_trends.png", dpi=300)
        plt.close()
        
    def _distribution_analysis(self, value_col: str, save_path: str):
        """Analiza distribuciones de variables clave."""
        print("\n📈 Análisis de Distribuciones:")
        
        # Distribución multimodal por métrica
        g = sns.FacetGrid(self.df, col='metric', col_wrap=3, sharey=False)
        g.map(sns.histplot, 'value', kde=True, bins=30)
        g.savefig(f"{save_path}/value_distributions.png", dpi=300)
        plt.close()
        
        # Boxplot interactivo
        fig = px.box(
            self.df,
            x='metric',
            y='value',
            color='fleet',
            title='Distribución por Métrica y Flota'
        )
        fig.write_html(f"{save_path}/interactive_boxplot.html")
        
    def _correlation_analysis(self, save_path: str):
        """Analiza relaciones entre variables numéricas."""
        if len(self.numeric_cols) < 2:
            return
            
        print("\n🧩 Análisis de Correlaciones:")
        
        # Matriz de correlación
        corr_matrix = self.df[self.numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación Numérica')
        plt.savefig(f"{save_path}/correlation_matrix.png", dpi=300)
        plt.close()
        
    def _outlier_detection(self, value_col: str, save_path: str):
        """Identifica y analiza valores atípicos."""
        print("\n🔍 Detección de Outliers:")
        
        # Identificación usando IQR
        Q1 = self.df[value_col].quantile(0.25)
        Q3 = self.df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[(self.df[value_col] < (Q1 - 1.5 * IQR)) | 
                   (self.df[value_col] > (Q3 + 1.5 * IQR))]
        
        if not outliers.empty:
            print(f"• Potenciales outliers detectados: {len(outliers):,} ({len(outliers)/len(self.df)*100:.2f}%)")
            print("  Distribución por métrica:")
            print(outliers['metric'].value_counts().to_string())
            
            # Guardar outliers para inspección
            outliers.to_csv(f"{save_path}/potential_outliers.csv", index=False)
        else:
            print("✅ No se detectaron outliers significativos")

# =============================================
# Bloque Principal Integrado
# =============================================
if __name__ == "__main__":
    try:
        # 1. Cargar datos
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        loader = ExploreDataLoader(project_root)
        print("\n🚚 Cargando datos...")
        raw_df = loader.load_data(
            truck_name='T-210',
            metrics=['fuel', 'rpm'],
            years=[2024, 2025]
        )
        
        # 2. Limpieza temporal
        print("\n🧹 Limpiando datos temporales...")
        clean_df = clean_and_analyze_temporal_data(raw_df)
        
        # 3. Análisis exploratorio
        print("\n🔍 Realizando análisis exploratorio...")
        eda = AdvancedEDA(clean_df)  # Usar el DataFrame limpio
        eda.perform_full_analysis(save_path='informes/eda')
        
        print("\n✅ Proceso completo finalizado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error en el proceso principal: {str(e)}")
        traceback.print_exc()
        sys.exit(1)