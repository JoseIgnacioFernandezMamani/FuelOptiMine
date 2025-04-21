import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración básica
st.set_page_config(
    page_title="OptiMine v0.1",
    layout="centered",
    page_icon="⛏️"
)

# Datos de ejemplo
@st.cache_data
def load_sample_data():
    return pd.DataFrame({
        'Fecha': pd.date_range(start='2024-01-01', periods=5),
        'Consumo_diesel': [120, 135, 128, 140, 132],
        'Tonelaje': [500, 480, 510, 490, 505]
    })

# --------------------------------------------------
# Configuración de Navegación
# --------------------------------------------------
PAGINAS = {
    "🏠 Inicio": "inicio",
    "📈 Modelo Lineal": "lineal",
    "🌳 XGBoost": "xgboost",
    "⚙️ Optimización MILP": "milp",
    "🤖 Asistente IA": "asistente",
    "⚙️ Ajustes": "ajustes"
}

# Sidebar: Selector de página
with st.sidebar:
    st.header("Navegación")
    pagina_seleccionada = st.radio(
        "Ir a:",
        options=list(PAGINAS.keys()),
        index=0,
        format_func=lambda x: x.split(" ")[-1]  # Muestra solo el nombre sin emoji
    )
    
    st.header("Configuración")
    fecha_inicio = st.date_input("Fecha inicial")
    fecha_fin = st.date_input("Fecha final")

# --------------------------------------------------
# Contenido Dinámico según Página
# --------------------------------------------------
df = load_sample_data()

if PAGINAS[pagina_seleccionada] == "inicio":
    st.title("📊 Monitor de Consumo - MSC")
    
    # Sección principal
    st.subheader("Datos de Ejemplo")
    st.dataframe(df)
    
    # Gráfico interactivo
    st.subheader("Tendencia de Consumo")
    fig = px.line(df, x='Fecha', y='Consumo_diesel', title='Consumo Diario')
    st.plotly_chart(fig, use_container_width=True)
    
    # Widget básico
    with st.expander("Diagnóstico Rápido"):
        consumo_promedio = df['Consumo_diesel'].mean()
        st.metric("Consumo Promedio", f"{consumo_promedio:.1f} L/día")

elif PAGINAS[pagina_seleccionada] == "Lineal":
    st.title("📈 Modelo de Regresión Lineal")
    
    # Controles específicos del modelo
    with st.expander("Parámetros Técnicos"):
        intercepto = st.slider("Intercepto", 50, 200, 100)
        coef_tonelaje = st.slider("Coef. Tonelaje", 0.1, 0.5, 0.3)
    
    # Gráfico de predicciones
    df['Predicción'] = intercepto + coef_tonelaje * df['Tonelaje']
    fig = px.scatter(df, x='Tonelaje', y='Consumo_diesel', trendline="ols")
    st.plotly_chart(fig)

elif PAGINAS[pagina_seleccionada] == "xgboost":
    st.title("🌳 Modelo XGBoost")
    
    # Simulación de importancia de características
    importancia = {
        'Tonelaje': 0.35,
        'Pendiente': 0.28,
        'Antigüedad': 0.17
    }
    
    fig = px.bar(
        x=list(importancia.values()),
        y=list(importancia.keys()),
        orientation='h',
        title="Importancia de Variables"
    )
    st.plotly_chart(fig)

elif PAGINAS[pagina_seleccionada] == "milp":
    st.title("⚙️ Optimización MILP")
    
    # Entradas para simulación
    camiones = st.number_input("N° Camiones", 1, 20, 5)
    st.button("Ejecutar Optimización")
    
    # Resultados simulados
    st.write("""
    **Resultados:**
    - Ruta óptima: Norte
    - Combustible estimado: 1200 L
    - Ahorro proyectado: 15%
    """)

elif PAGINAS[pagina_seleccionada] == "asistente":
    st.title("🤖 Asistente de IA")
    
    # Chat básico
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Haz tu pregunta"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "Respuesta simulada"})
        st.rerun()

elif PAGINAS[pagina_seleccionada] == "ajustes":
    st.title("⚙️ Configuraciones")
    
    with st.form("config_form"):
        st.subheader("Parámetros de Conexión")
        servidor = st.text_input("Servidor BD", "sql.msc.local")
        usuario = st.text_input("Usuario")
        contraseña = st.text_input("Contraseña", type="password")
        
        if st.form_submit_button("Guardar"):
            st.success("Configuración actualizada")











            class InteractiveDataExplorer:
    def __init__(self, loader: ExploreDataLoader):
        self.loader = loader
        self.output = Output()
        self._is_running = False
        self._current_process = None
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura los componentes de la interfaz de usuario"""
        # Widgets de selección
        self.truck_dropdown = widgets.Dropdown(
            options=self.loader.available_trucks,
            description='Camión:',
            style={'description_width': 'initial'}
        )
        
        self.metrics_selector = widgets.SelectMultiple(
            options=['fuel', 'rpm', 'cycle', 'time_model'],
            value=['rpm'],
            description='Métricas:',
            rows=4,
            style={'description_width': 'initial'}
        )
        
        self.years_selector = widgets.SelectMultiple(
            options=[str(y) for y in range(2024, 2031)],
            value=['2024'],
            description='Años:',
            rows=7,
            style={'description_width': 'initial'}
        )
        
        # Botones de control
        self.start_btn = widgets.Button(
            description='▶️ Iniciar',
            button_style='success',
            tooltip='Iniciar búsqueda y visualización'
        )
        
        self.stop_btn = widgets.Button(
            description='⏹ Detener',
            button_style='danger',
            tooltip='Detener proceso actual',
            disabled=True
        )
        
        self.cancel_btn = widgets.Button(
            description='⏹ Cancelar',
            button_style='warning',
            tooltip='Cancelar operación'
        )
        
        # Indicador de estado
        self.status = widgets.HTML(
            value="🟢 Listo",
            style={'font_size': '16px'}
        )
        
        # Diseño
        self.controls = VBox([
            self.truck_dropdown,
            self.metrics_selector,
            self.years_selector,
            HBox([self.start_btn, self.stop_btn, self.cancel_btn, self.status]),
            self.output
        ])
        
        # Eventos
        self.start_btn.on_click(self._start_search)
        self.stop_btn.on_click(self._stop_search)
        self.cancel_btn.on_click(self._cancel_search)
        
    def _update_ui_state(self, running: bool):
        """Actualiza el estado de los controles UI"""
        self.start_btn.disabled = running
        self.stop_btn.disabled = not running
        self.cancel_btn.disabled = not running
        self.status.value = "🟠 Procesando..." if running else "🟢 Listo"
        
    def _start_search(self, btn):
        """Inicia el proceso de búsqueda y visualización"""
        if self._is_running:
            return
            
        self._is_running = True
        self._update_ui_state(True)
        
        # Ejecutar en un hilo separado para no bloquear la UI
        self._current_process = threading.Thread(target=self._execute_search)
        self._current_process.start()
        
    def _execute_search(self):
        """Ejecuta la búsqueda real con logging detallado"""
        try:
            with self.output:
                clear_output(wait=True)
                print("🔄 Iniciando proceso de búsqueda...\n")
                
                # 1. Generar patrones
                patterns = self.loader._generate_file_patterns(
                    self.truck_dropdown.value,
                    self.metrics_selector.value,
                    self.years_selector.value
                )
                print("🔍 Patrones generados:")
                for idx, pattern in enumerate(patterns, 1):
                    print(f"  {idx}. {pattern}")
                
                # 2. Buscar archivos
                print("\n📂 Búsqueda de archivos...")
                files = self.loader._find_matching_files(patterns)
                if not files:
                    raise FileNotFoundError("No se encontraron archivos")
                    
                print("\n✅ Archivos encontrados:")
                for idx, file in enumerate(files, 1):
                    print(f"  {idx}. {os.path.basename(file)}")
                
                # 3. Cargar datos
                print("\n⏳ Cargando datos...")
                data = self.loader._load_and_merge_data(files)
                
                # 4. Resumen final
                print("\n📊 Resumen de datos cargados:")
                print(f"- Total registros: {len(data):,}")
                print(f"- Métricas cargadas: {', '.join(data['metric'].unique())}")
                print(f"- Rango temporal: {data['timestamp'].min()} a {data['timestamp'].max()}")
                
                # 5. Generar visualización
                print("\n🎨 Generando visualización...")
                self._plot_data(data)
                print("\n✅ Proceso completado con éxito!")
                
        except Exception as e:
            print(f"\n🚨 Error crítico: {str(e)}")
        finally:
            self._is_running = False
            self._update_ui_state(False)
        
    def _stop_search(self, btn):
        """Detiene el proceso actual"""
        if self._is_running:
            with self.output:
                print("⏹ Proceso detenido por el usuario")
            self._is_running = False
            self._update_ui_state(False)
        
    def _cancel_search(self, btn):
        """Cancela y reinicia los parámetros"""
        with self.output:
            clear_output()
            print("⚠️ Operación cancelada")
        self._is_running = False
        self._update_ui_state(False)
      
    def _plot_data(self, data: pd.DataFrame):
        """Crea la visualización interactiva con Plotly"""
        fig = go.Figure()
        
        # Agrupar por métrica y año
        grouped = data.groupby(['metric', pd.to_datetime(data['timestamp']).dt.year])
        
        for (metric, year), group in grouped:
            fig.add_trace(go.Scatter(
                x=group['timestamp'],
                y=group['value'],
                mode='lines+markers',
                name=f"{metric.upper()} {year}",
                hovertemplate="%{y:.2f}<extra>%{x|%Y-%m-%d %H:%M}</extra>"
            ))
        
        fig.update_layout(
            title=f"Datos del camión {self.truck_dropdown.value}",
            xaxis_title='Fecha y Hora',
            yaxis_title='Valor',
            hovermode='x unified',
            height=600,
            template='plotly_dark'
        )
        
        fig.show()

    def show(self):
        """Muestra la interfaz de usuario"""
        display(self.controls)

# Configuración y ejecución
if __name__ == "__main__":
    project_root = os.path.normpath(os.path.join(os.getcwd(), '..'))
    loader = ExploreDataLoader(project_root)
    explorer = InteractiveDataExplorer(loader)
    explorer.show()
    

























# =============================================
# Bloque 6 - Visualización Interactiva Completa
# =============================================
import plotly.graph_objects as go
from ipywidgets import interact, widgets
from IPython.display import display

# 1. Preparación de datos
def prepare_data(df):
    # Convertir a datetime si no está en el formato correcto
    if not pd.api.types.is_datetime64_any_dtype(df['created_at_local']):
        df['created_at_local'] = pd.to_datetime(df['created_at_local'], errors='coerce')
    
    # Crear columna año-mes y día
    df['year_month'] = df['created_at_local'].dt.strftime('%Y-%m')
    df['day'] = df['created_at_local'].dt.day
    
    # Eliminar registros sin fecha válida
    return df.dropna(subset=['created_at_local'])

# 2. Función de visualización mejorada
def interactive_daily_histogram(clean_df):
    # Widget para selección de mes
    month_selector = widgets.Dropdown(
        options=sorted(clean_df['year_month'].unique()),
        description='Seleccionar Mes:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )

    # Widget para selección de métrica
    metric_selector = widgets.Dropdown(
        options=clean_df['metric'].unique().tolist(),
        description='Seleccionar Métrica:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )

    @interact(Mes=month_selector, Métrica=metric_selector)
    def update_plot(Mes, Métrica):
        # Filtrar datos
        filtered_data = clean_df[
            (clean_df['year_month'] == Mes) & 
            (clean_df['metric'] == Métrica)
        ]
        
        # Crear conteos diarios
        days = list(range(1, 32))
        daily_counts = filtered_data.groupby('day').size().reindex(days, fill_value=0)

        # Crear gráfico interactivo
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=daily_counts.index,
            y=daily_counts.values,
            marker=dict(
                color=daily_counts.values,
                colorscale='Tealgrn',
                cmin=0,
                cmax=daily_counts.max(),
                colorbar=dict(title='Registros')
        ))

        # Líneas de referencia
        avg = daily_counts.mean()
        max_day = daily_counts.idxmax()
        
        fig.update_layout(
            title=f'{Métrica} - Registros Diarios: {Mes}',
            xaxis=dict(
                title='Día del Mes',
                tickvals=days,
                tickangle=45),
            yaxis=dict(title='Total de Registros'),
            template='plotly_white',
            height=500,
            annotations=[
                dict(
                    x=max_day,
                    y=daily_counts[max_day],
                    text="Máximo",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            ]
        )

        fig.add_hline(y=avg, 
                     line_dash="dot", 
                     line_color="orange",
                     annotation_text=f'Promedio: {avg:.1f}')

        fig.show()

    # Mostrar controles
    display(month_selector)
    display(metric_selector)

# 3. Ejecutar la visualización con los datos
clean_df_prepared = prepare_data(clean_df.copy())
interactive_daily_histogram(clean_df_prepared)