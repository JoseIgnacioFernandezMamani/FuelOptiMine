import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# Crear histograma
fig = px.histogram(
    df_events,
    x='SensorHL',
    y='cantidad',
    title='Eventos capturados por el sensor por hora del día',
    nbins=24,
    color_discrete_sequence=['#636efa'],
    text_auto=True,
    barmode='group'
)

# Ordenar los datos para identificar los 4 más grandes y 4 más pequeños
df_sorted = df_events.sort_values(by='cantidad')
lowest = df_sorted.head(4)
highest = df_sorted.tail(4)

# Añadir marcadores para los más pequeños y más grandes valores del histograma
fig.add_trace(go.Scatter(
    x=lowest['SensorHL'],
    y=lowest['cantidad'],
    mode='markers',
    marker=dict(color='red', size=10, symbol='circle'),
    name='Más pequeños'
))

fig.add_trace(go.Scatter(
    x=highest['SensorHL'],
    y=highest['cantidad'],
    mode='markers',
    marker=dict(color='green', size=10, symbol='circle'),
    name='Más grandes'
))

# Personalización del layout
fig.update_layout(
    xaxis_title='Hora del día',
    yaxis_title='Cantidad de eventos',
    xaxis=dict(
        tick0=0,
        dtick=1,
        tickformat='%H'
    ),
    yaxis=dict(
        tickformat=".0f",
        hoverformat=".0f"
    )
)

fig.show()



# libreria para generare graficos en python
import plotly.express as px

fig = px.histogram(
    df_events_per_minute,
    x='SensorHL',
    y='cantidad',
    title='Eventos capturados por el sensor por minuto del día',
    nbins=1440,  # Fuerza 24 bins (1 por hora)
    color_discrete_sequence=['#636efa'],  # Color personalizado
    barmode='group'
)

# Ordenar los datos para identificar los 4 más grandes y 4 más pequeños
df_events_per_minute_sorted = df_events_per_minute.sort_values(by='cantidad')
lowest = df_events_per_minute_sorted.head(4)
highest = df_events_per_minute_sorted.tail(4)

# Añadir marcadores para los más pequeños y más grandes
fig.add_trace(go.Scatter(
    x=lowest['SensorHL'],
    y=lowest['cantidad'],
    mode='markers',
    marker=dict(color='red', size=10, symbol='circle'),
    name='Más pequeños'
))

fig.add_trace(go.Scatter(
    x=highest['SensorHL'],
    y=highest['cantidad'],
    mode='markers',
    marker=dict(color='green', size=10, symbol='circle'),
    name='Más grandes'
))

# Personalizacion del layout
fig.update_layout(
    xaxis_title='minuto del día',
    yaxis_title='Cantidad de eventos',
    xaxis=dict(
        showticklabels=False,  # Oculta etiquetas de horas
    ),
    yaxis=dict(
        tickformat=".0f",
        hoverformat=".0f"
    )
)
fig.show()




from ipywidgets import interact, widgets
import plotly.express as px

def prepare_daily_data(df: pd.DataFrame, truck_id: str) -> pd.DataFrame:
    # Filtrar por camión
    df = df[df['Truck'] == truck_id].copy()
    
    # Asegurar zona horaria
    df['fecha'] = df['TimeStamp'].dt.tz_localize('UTC').dt.tz_convert('America/Santiago')
    
    # Extraer componentes
    df['mes'] = df['fecha'].dt.month_name(locale='es')
    df['dia_mes'] = df['fecha'].dt.day
    df['año'] = df['fecha'].dt.year

    # Agrupar por día
    daily_counts = df.groupby(['año', 'mes', 'dia_mes']).size().reset_index(name='conteo')

    # Orden de meses en español
    month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    daily_counts['mes'] = pd.Categorical(daily_counts['mes'], categories=month_order, ordered=True)

    return daily_counts.sort_values(['año', 'mes', 'dia_mes'])


def plot_daily_histogram_interactive(df_sensor: pd.DataFrame, truck_id: str):
    # Prepara datos
    hist_data = prepare_daily_data(df_sensor, truck_id)
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
            title=f'Registros diarios de {truck_id} - {Mes}',
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










# =============================================
# 6. Histograma Interactivo de Registros Diarios por Mes para fuel
# =============================================
from ipywidgets import interact, widgets
import plotly.express as px

# Preparar datos para histograma
def prepare_histogram_data(df):
    # Extraer componentes de fecha
    df = df[df['metric'] == 'fuel'].copy()
    df['fecha'] = df['created_at_local'].dt.tz_convert('America/Santiago')
    df['mes'] = df['fecha'].dt.month_name(locale='es')
    df['dia_mes'] = df['fecha'].dt.day
    df['año'] = df['fecha'].dt.year
    
    # Agregar conteos diarios
    daily_counts = df.groupby(['año', 'mes', 'dia_mes']).size().reset_index(name='conteo')
    
    # Ordenar meses cronológicamente
    month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    daily_counts['mes'] = pd.Categorical(daily_counts['mes'], categories=month_order, ordered=True)
    
    return daily_counts.sort_values(['año', 'mes', 'dia_mes'])

# Generar datos para el histograma
hist_data = prepare_histogram_data(clean_df)

# Widget interactivo
@interact(
    Mes=widgets.Dropdown(
        options=hist_data['mes'].unique(),
        description='Seleccionar Mes:',
        style={'description_width': 'initial'}
    )
)
def plot_daily_histogram(Mes):
    # Filtrar datos
    filtered = hist_data[hist_data['mes'] == Mes]
    
    # Crear gráfico
    fig = px.bar(
        filtered,
        x='dia_mes',
        y='conteo',
        color='año',
        barmode='group',
        title=f'Registros Diarios - {Mes}',
        labels={'dia_mes': 'Día del Mes', 'conteo': 'Número de Registros'},
        template='plotly_white'
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[0.5, 31.5]
        ),
        hovermode='x unified',
        height=500
    )
    
    # Añadir línea de promedio
    avg = filtered['conteo'].mean()
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="red",
        annotation_text=f'Promedio: {avg:.1f}',
        annotation_position="top right"
    )
    
    fig.show()




# =============================================
# 6. Histograma Interactivo de Registros Diarios por Mes (Solo RPM)
# =============================================
from ipywidgets import interact, widgets
import plotly.express as px

# Preparar datos para histograma (solo RPM)
def prepare_rpm_histogram_data(df):
    # Filtrar solo datos de RPM
    rpm_df = df[df['metric'] == 'rpm'].copy()
    
    # Extraer componentes de fecha
    rpm_df['fecha'] = rpm_df['created_at_local'].dt.tz_convert('America/Santiago')
    rpm_df['mes'] = rpm_df['fecha'].dt.month_name(locale='es')
    rpm_df['dia_mes'] = rpm_df['fecha'].dt.day
    rpm_df['año'] = rpm_df['fecha'].dt.year
    
    # Agregar conteos diarios
    daily_counts = rpm_df.groupby(['año', 'mes', 'dia_mes']).size().reset_index(name='conteo')
    
    # Ordenar meses cronológicamente
    month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    daily_counts['mes'] = pd.Categorical(daily_counts['mes'], categories=month_order, ordered=True)
    
    return daily_counts.sort_values(['año', 'mes', 'dia_mes'])

# Generar datos para el histograma (solo RPM)
rpm_hist_data = prepare_rpm_histogram_data(clean_df)

# Widget interactivo para RPM
@interact(
    Mes=widgets.Dropdown(
        options=rpm_hist_data['mes'].unique(),
        description='Seleccionar Mes (RPM):',
        style={'description_width': 'initial'}
    )
)
def plot_rpm_daily_histogram(Mes):
    # Filtrar datos
    filtered = rpm_hist_data[rpm_hist_data['mes'] == Mes]
    
    # Crear gráfico
    fig = px.bar(
        filtered,
        x='dia_mes',
        y='conteo',
        color='año',
        barmode='group',
        title=f'Registros Diarios de RPM - {Mes}',
        labels={'dia_mes': 'Día del Mes', 'conteo': 'Número de Registros RPM'},
        template='plotly_white'
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[0.5, 31.5]
        ),
        hovermode='x unified',
        height=500
    )
    
    # Añadir línea de promedio
    avg = filtered['conteo'].mean()
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="red",
        annotation_text=f'Promedio: {avg:.1f}',
        annotation_position="top right"
    )
    
    fig.show()