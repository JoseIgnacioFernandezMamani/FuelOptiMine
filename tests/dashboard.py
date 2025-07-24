# dashboard.py
import os
import streamlit as st
import plotly.graph_objects as go
import clickhouse_connect
import queries  # Asegúrate que está en el mismo directorio
import polars as pl

# Conexión a ClickHouse
CH_CONFIG = {
    "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
    "port": int(os.getenv("CLICKHOUSE_NATIVE_PORT", 8123)),
    "username": os.getenv("CLICKHOUSE_USER", "default"),
    "password": os.getenv("CLICKHOUSE_PASSWORD", "password"),
    "database": os.getenv("CLICKHOUSE_DB", "fuel_optimine"),
    "compress": True,
    "send_receive_timeout": 300,
}

client = clickhouse_connect.get_client(**CH_CONFIG)

# Página principal
st.set_page_config(layout="wide")
st.title("⛽ Dashboard de Sensor de Combustible - FuelOptiMine")

# Consultar datos diarios
df_resampled = client.query_df(queries.fuel_level_timeseries)
df_resampled = pl.from_pandas(df_resampled)

df_raw_today = client.query_df(queries.fuel_level_by_day)
df_raw_today = pl.from_pandas(df_raw_today)

tab1, tab2 = st.tabs(["📊 Serie Temporal (Promedios)", "📈 Valores Crudos de Hoy"])

with tab1:
    st.header("📊 Promedio de nivel de combustible por día")

    if df_resampled.is_empty():
        st.warning("No hay datos disponibles para la serie temporal.")
    else:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_resampled["day"].to_numpy(),
                y=df_resampled["avg_fuel_level"].to_numpy(),
                mode="lines+markers",
                name="Promedio",
                line=dict(color="#1f77b4"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_resampled["day"].to_numpy(),
                y=df_resampled["max_fuel_level"].to_numpy(),
                mode="lines",
                name="Máximo",
                line=dict(width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_resampled["day"].to_numpy(),
                y=df_resampled["min_fuel_level"].to_numpy(),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.2)",
                name="Rango Min-Max",
            )
        )

        fig.update_layout(
            title="Promedio de FuelLevelLiters por Día",
            xaxis_title="Fecha",
            yaxis_title="Litros",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Días registrados", len(df_resampled))
        with col2:
            st.metric(
                "Promedio general", f"{df_resampled['avg_fuel_level'].mean():.2f} L"
            )
        with col3:
            st.metric(
                "Máximo registrado", f"{df_resampled['max_fuel_level'].max():.2f} L"
            )

with tab2:
    st.header("📈 Serie Temporal de Hoy")

    if df_raw_today.is_empty():
        st.warning("No hay datos para hoy.")
    else:
        fig2 = go.Figure()

        fig2.add_trace(
            go.Scattergl(
                x=df_raw_today["TimeStamp"].to_numpy(),
                y=df_raw_today["FuelLevelLiters"].to_numpy(),
                mode="lines+markers",
                name="FuelLevelLiters",
                line=dict(color="#2ca02c", width=2),
            )
        )

        fig2.update_layout(
            title="Niveles de Combustible Hoy",
            xaxis_title="Hora",
            yaxis_title="Litros",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig2, use_container_width=True)
