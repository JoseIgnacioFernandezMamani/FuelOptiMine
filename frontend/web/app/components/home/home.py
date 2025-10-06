import streamlit as st
import polars as pl
import plotly.graph_objects as go
import numpy as np
from etl_core.load.utils import create_client, CH_CONFIG
from model.predictive.mlflow_config import TRUCK_IDS


# load from clickhouse
@st.cache_data(ttl=300)
def load_sensor_data(truck_id: str):
    try:
        client = create_client(CH_CONFIG)
        query = f"""
        SELECT
            TimeStamp,
            TimeStampIni,
            TimeStampFin,
            TimeStamp_tm,
            StageType,
            Status,
            Category,
            Event,
            FuelLevelLiters,
            SpeedAvg,
            ValidFuel
        FROM xgboost_fuel
        WHERE Equipment = '{truck_id}'
        ORDER BY TimeStamp
        """
        pandas_df = client.query_df(query)
        if pandas_df.empty:
            return None
        return pl.from_pandas(pandas_df).sort("TimeStamp")
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        return None


# ---------------------------------------------------------
# 🔧 Función para Min-Max Scaling
# ---------------------------------------------------------
def min_max_scaler(data, feature_range=(0, 1)):
    """Aplica Min-Max scaling a los datos"""
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:  # Evitar división por cero
        return np.zeros_like(data)

    scaled = (data - min_val) / (max_val - min_val)
    # Escalar al rango especificado
    scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    return scaled, min_val, max_val


# ---------------------------------------------------------
# 📊 Interfaz principal
# ---------------------------------------------------------
def show():
    with st.sidebar:
        st.header("🚚 Selección de Camión")
        truck_id = st.selectbox(
            "ID del Camión:",
            options=TRUCK_IDS,
            index=0,
        )

    st.title(f"📊 Análisis de Sensores - {truck_id}")

    # Cargar datos
    with st.spinner("Cargando datos del sensor..."):
        df = load_sensor_data(truck_id)
        if df is None or df.is_empty():
            st.error("❌ No se encontraron datos para este camión.")
            return

        min_date = df.select(pl.col("TimeStamp").dt.date()).min().item()
        max_date = df.select(pl.col("TimeStamp").dt.date()).max().item()

        # Select analysis period
        period_option = st.selectbox(
            "Seleccionar período de análisis",
            ["Dia", "Semana", "Mes", "Trimestre"],
            index=0,
        )

        # Configuración de variables
        variables_config = {
            "FuelLevelLiters": {
                "label": "Nivel de Combustible",
                "color": "#1f77b4",
                "unit": "L",
            },
            "SpeedAvg": {
                "label": "Velocidad Promedio",
                "color": "#2ca02c",
                "unit": "km/h",
            },
        }

        if period_option == "Dia":
            selected_date = st.date_input(
                "Seleccionar fecha para análisis temporal",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )
            filtered_data = df.filter(pl.col("TimeStamp").dt.date() == selected_date)

            if not filtered_data.is_empty():
                # Obtener datos originales
                x_axis = filtered_data["TimeStamp"].to_numpy()
                fuel_data = filtered_data["FuelLevelLiters"].to_numpy()
                speed_data = filtered_data["SpeedAvg"].to_numpy()
                valid_fuel_data = filtered_data["ValidFuel"].to_numpy()

                # Aplicar Min-Max Scaling
                fuel_scaled, fuel_min, fuel_max = min_max_scaler(fuel_data)
                speed_scaled, speed_min, speed_max = min_max_scaler(speed_data)

                # Crear figura
                fig = go.Figure()

                # Agregar combustible escalado
                fig.add_trace(
                    go.Scattergl(
                        x=x_axis,
                        y=fuel_scaled,
                        mode="lines",
                        name="Nivel de Combustible (escalado)",
                        line=dict(
                            color=variables_config["FuelLevelLiters"]["color"], width=3
                        ),
                        hovertemplate=(
                            "Fecha: %{x}<br>"
                            + "Combustible: %{customdata:.1f} L<br>"
                            + "Escalado: %{y:.3f}<extra></extra>"
                        ),
                        customdata=fuel_data,
                    )
                )

                # Agregar velocidad escalada
                fig.add_trace(
                    go.Scattergl(
                        x=x_axis,
                        y=speed_scaled,
                        mode="lines",
                        name="Velocidad Promedio (escalado)",
                        line=dict(color=variables_config["SpeedAvg"]["color"], width=3),
                        hovertemplate=(
                            "Fecha: %{x}<br>"
                            + "Velocidad: %{customdata:.1f} km/h<br>"
                            + "Escalado: %{y:.3f}<extra></extra>"
                        ),
                        customdata=speed_data,
                    )
                )
                # Agregar líneas verticales rojas donde ValidFuel > 0
                valid_fuel_events = []
                for i, (timestamp, valid_fuel) in enumerate(
                    zip(x_axis, valid_fuel_data)
                ):
                    if valid_fuel > 0:
                        valid_fuel_events.append(timestamp)
                        fig.add_shape(
                            type="line",
                            x0=timestamp,
                            x1=timestamp,
                            y0=0,
                            y1=1,
                            line=dict(
                                color="red",
                                width=2,
                                dash="dash",
                            ),
                            opacity=0.7,
                        )

                # Configurar layout
                fig.update_layout(
                    title=f"Nivel de Combustible y Velocidad (Escalados) - {selected_date}",
                    xaxis_title="Hora del Día",
                    yaxis_title="Valor Escalado (0-1)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    yaxis=dict(range=[0, 1.1]),  # Rango fijo para mejor visualización
                )

                # Reemplaza esta sección del código:

                # Agregar timeline de StageType (TimeStampIni - TimeStampFin)
                stage_events = (
                    filtered_data.filter(
                        pl.col("TimeStampIni").is_not_null()
                        & pl.col("TimeStampFin").is_not_null()
                    )
                    .select(["TimeStampIni", "TimeStampFin", "StageType"])
                    .unique()
                )

                if not stage_events.is_empty():
                    for event in stage_events.iter_rows(named=True):
                        # Agregar línea vertical para el inicio (TimeStampIni)
                        fig.add_shape(
                            type="line",
                            x0=event["TimeStampIni"],
                            x1=event["TimeStampIni"],
                            y0=0,
                            y1=0.04,
                            line=dict(
                                color="blue",
                                width=2,
                            ),
                            opacity=0.7,
                        )
                        # Agregar punto invisible para el hover del inicio
                        fig.add_trace(
                            go.Scatter(
                                x=[event["TimeStampIni"]],
                                y=[0.02],
                                mode="markers",
                                marker=dict(size=8, color="blue", opacity=0),
                                hovertemplate=(
                                    f"StageType: {event['StageType']}<br>"
                                    f"Inicio: {event['TimeStampIni']}<br>"
                                    f"Tipo: Límite Inicial<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=False,
                                name=f"Stage Start: {event['StageType']}",
                            )
                        )

                        # Agregar línea vertical para el fin (TimeStampFin)
                        fig.add_shape(
                            type="line",
                            x0=event["TimeStampFin"],
                            x1=event["TimeStampFin"],
                            y0=0,
                            y1=0.04,
                            line=dict(
                                color="purple",
                                width=2,
                            ),
                            opacity=0.7,
                        )
                        # Agregar punto invisible para el hover del fin
                        fig.add_trace(
                            go.Scatter(
                                x=[event["TimeStampFin"]],
                                y=[0.02],
                                mode="markers",
                                marker=dict(size=8, color="purple", opacity=0),
                                hovertemplate=(
                                    f"StageType: {event['StageType']}<br>"
                                    f"Fin: {event['TimeStampFin']}<br>"
                                    f"Tipo: Límite Final<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=False,
                                name=f"Stage End: {event['StageType']}",
                            )
                        )

                # Agregar eventos de TimeStamp_tm
                tm_events = (
                    filtered_data.filter(pl.col("TimeStamp_tm").is_not_null())
                    .select(["TimeStamp_tm", "Status", "Category", "Event"])
                    .unique()
                )

                if not tm_events.is_empty():
                    for event in tm_events.iter_rows(named=True):
                        fig.add_shape(
                            type="line",
                            x0=event["TimeStamp_tm"],
                            x1=event["TimeStamp_tm"],
                            y0=0,
                            y1=0.1,  # Línea más larga para TimeStamp_tm
                            line=dict(
                                color="orange",
                                width=3,
                                dash="dot",
                            ),
                            opacity=0.8,
                        )
                        # Agregar hover personalizado para mostrar más información
                        fig.add_trace(
                            go.Scatter(
                                x=[event["TimeStamp_tm"]],
                                y=[0.05],
                                mode="markers",
                                marker=dict(size=8, color="orange", opacity=0),
                                hovertemplate=(
                                    f"TimeStamp_tm: {event['TimeStamp_tm']}<br>"
                                    f"Status: {event['Status']}<br>"
                                    f"Category: {event['Category']}<br>"
                                    f"Event: {event['Event']}<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=False,
                                name=f"{event['Status']} - {event['Category']}",
                            )
                        )

                # Agregar leyenda explicativa si hay eventos
                if not stage_events.is_empty() or not tm_events.is_empty():
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(size=10, color="lightblue"),
                            name="StageType Events",
                            showlegend=True,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(size=10, color="orange"),
                            name="Status/Category/Events",
                            showlegend=True,
                        )
                    )
                st.plotly_chart(fig, use_container_width=True)

                # Mostrar información de escalado
                st.info(
                    "📊 **Nota:** Ambas variables han sido escaladas al rango 0-1 usando Min-Max Scaling para mejor visualización"
                )
                # Mostrar estadísticas de eventos ValidFuel
                if len(valid_fuel_events) > 0:
                    st.success(
                        f"🔴 **Eventos de ValidFuel > 0 detectados:** {len(valid_fuel_events)} eventos"
                    )

                    # Mostrar detalles de los eventos
                    with st.expander("📋 Ver detalles de eventos ValidFuel"):
                        event_details = []
                        for i, event_time in enumerate(valid_fuel_events):
                            # Encontrar la fila correspondiente al evento
                            event_row = filtered_data.filter(
                                pl.col("TimeStamp") == event_time
                            )
                            if not event_row.is_empty():
                                fuel_value = event_row["FuelLevelLiters"].item()
                                speed_value = event_row["SpeedAvg"].item()
                                event_details.append(
                                    {
                                        "Número": i + 1,
                                        "Timestamp": event_time,
                                        "Combustible (L)": f"{fuel_value:.1f}",
                                        "Velocidad (km/h)": f"{speed_value:.1f}",
                                    }
                                )

                        if event_details:
                            # Crear DataFrame para mostrar
                            events_df = pl.DataFrame(event_details)
                            st.dataframe(events_df, use_container_width=True)
                else:
                    st.warning(
                        "⚠️ No se detectaron eventos con ValidFuel > 0 en esta fecha"
                    )

                # Mostrar estadísticas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Combustible Min-Max", f"{fuel_min:.0f}-{fuel_max:.0f} L")
                with col2:
                    st.metric(
                        "Velocidad Min-Max", f"{speed_min:.1f}-{speed_max:.1f} km/h"
                    )
                with col3:
                    st.metric("Combustible Promedio", f"{fuel_data.mean():.1f} L")
                with col4:
                    st.metric("Velocidad Promedio", f"{speed_data.mean():.1f} km/h")

            else:
                st.warning(f"No hay datos para el {selected_date}")

        else:
            # Análisis con resampling para períodos mayores
            if not df.is_empty():
                df_sorted = df.sort("TimeStamp")

                # Configurar el período de resampling
                if period_option == "Semana":
                    resample_period = "1w"
                    title_suffix = "por Semana"
                    x_title = "Semanas"
                elif period_option == "Mes":
                    resample_period = "1mo"
                    title_suffix = "por Mes"
                    x_title = "Meses"
                elif period_option == "Trimestre":
                    resample_period = "1q"
                    title_suffix = "por Trimestre"
                    x_title = "Trimestres"

                # Realizar resampling
                resampled_data = (
                    df_sorted.group_by_dynamic("TimeStamp", every=resample_period)
                    .agg(
                        [
                            pl.col("FuelLevelLiters").mean().alias("mean_fuel"),
                            pl.col("FuelLevelLiters").std().alias("std_fuel"),
                            pl.col("SpeedAvg").mean().alias("mean_speed"),
                            pl.col("SpeedAvg").std().alias("std_speed"),
                            pl.col("FuelLevelLiters").count().alias("count"),
                        ]
                    )
                    .filter(pl.col("count") > 0)
                    .sort("TimeStamp")
                )

                if not resampled_data.is_empty():
                    # Obtener datos resampleados
                    x_axis = resampled_data["TimeStamp"].to_numpy()
                    mean_fuel = resampled_data["mean_fuel"].to_numpy()
                    mean_speed = resampled_data["mean_speed"].to_numpy()

                    # Aplicar scaling a los promedios
                    fuel_scaled, fuel_min, fuel_max = min_max_scaler(mean_fuel)
                    speed_scaled, speed_min, speed_max = min_max_scaler(mean_speed)

                    # Crear figura
                    fig = go.Figure()

                    # Agregar combustible escalado
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=fuel_scaled,
                            mode="lines+markers",
                            name="Combustible Promedio (escalado)",
                            line=dict(
                                color=variables_config["FuelLevelLiters"]["color"],
                                width=3,
                            ),
                            marker=dict(size=6),
                            hovertemplate=(
                                "Período: %{x}<br>"
                                + "Combustible: %{customdata:.1f} L<br>"
                                + "Escalado: %{y:.3f}<extra></extra>"
                            ),
                            customdata=mean_fuel,
                        )
                    )

                    # Agregar velocidad escalada
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=speed_scaled,
                            mode="lines+markers",
                            name="Velocidad Promedio (escalado)",
                            line=dict(
                                color=variables_config["SpeedAvg"]["color"], width=3
                            ),
                            marker=dict(size=6),
                            hovertemplate=(
                                "Período: %{x}<br>"
                                + "Velocidad: %{customdata:.1f} km/h<br>"
                                + "Escalado: %{y:.3f}<extra></extra>"
                            ),
                            customdata=mean_speed,
                        )
                    )

                    # Configurar layout
                    fig.update_layout(
                        title=f"Combustible y Velocidad Promedio (Escalados) {title_suffix}",
                        xaxis_title=x_title,
                        yaxis_title="Valor Escalado (0-1)",
                        hovermode="x unified",
                        template="plotly_white",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        yaxis=dict(range=[0, 1.1]),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.info(
                        "📊 **Nota:** Los promedios han sido escalados al rango 0-1 usando Min-Max Scaling"
                    )

                    # Mostrar métricas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Períodos analizados", len(resampled_data))
                    with col2:
                        st.metric("Combustible Avg", f"{mean_fuel.mean():.1f} L")
                    with col3:
                        st.metric("Velocidad Avg", f"{mean_speed.mean():.1f} km/h")
                    with col4:
                        st.metric("Datos totales", f"{resampled_data['count'].sum():,}")

                else:
                    st.warning(
                        f"No hay datos suficientes para el análisis {title_suffix.lower()}"
                    )
            else:
                st.warning("No hay datos disponibles para el análisis")


if __name__ == "__main__":
    show()
