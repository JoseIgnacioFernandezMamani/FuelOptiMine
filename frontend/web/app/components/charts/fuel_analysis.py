from analitycs.fuel_analysis import FuelAnalysisOptimized

# app.py
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
from analitycs.fuel_analysis import FuelAnalysisOptimized


def main():
    st.title("📊 Análisis Avanzado de Combustible")

    analyzer = FuelAnalysisOptimized()
    stats = analyzer.get_temporal_gaps_stats()
    min_date, max_date = analyzer.get_date_range()

    # Configurar fecha por defecto
    default_date = (
        datetime(2025, 2, 2) if datetime(2025, 2, 2) >= min_date else min_date
    )

    # Selector de fecha
    st.sidebar.header("Parámetros de Análisis")
    selected_date = st.sidebar.date_input(
        "Seleccione un día:",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )

    # Obtener datos para visualización
    selected_datetime = datetime.combine(selected_date, datetime.min.time())
    df_viz = analyzer.get_visualization_data(selected_datetime)

    # Mostrar estadísticas clave
    st.header("Métricas Clave")
    cols = st.columns(3)
    cols[0].metric("Umbral Gap Normal", f"{stats['umbral_normal_gap']} segundos")
    cols[1].metric("Gaps Inusuales", stats["gaps_inusuales"])
    cols[2].metric("Recargas Detectadas", stats["recargas_detectadas"])

    # Gráfico interactivo
    st.header("Análisis Detallado por Turno")

    if df_viz is not None and not df_viz.is_empty():
        df_pd = df_viz.to_pandas()
        start_time = selected_datetime.replace(hour=7, minute=0, second=0)
        end_time = start_time + timedelta(days=1)

        # Crear figura
        fig = px.scatter(
            df_pd,
            x="TimeStamp",
            y="FuelLevelLiters",
            color="PosibleRecargaReal",
            color_discrete_map={True: "red", False: "blue"},
            size="RecordDuration",
            hover_data=["RecordDuration", "DeltaFuel"],
            title=f"Análisis de Combustible: {start_time.strftime('%d/%m %H:%M')} - {end_time.strftime('%d/%m %H:%M')}",
        )

        # Personalización avanzada
        fig.update_traces(
            marker=dict(opacity=0.8, line=dict(width=1, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )

        fig.update_layout(
            xaxis=dict(
                range=[start_time, end_time],
                tickformat="%H:%M\n%d/%m",
                title="Hora del Día",
            ),
            yaxis=dict(title="Nivel de Combustible (%)"),
            legend=dict(title="Posible Recarga"),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            height=600,
        )

        # Añadir línea de tiempo
        fig.add_shape(
            type="rect",
            x0=start_time.replace(hour=19),
            y0=0,
            x1=start_time + timedelta(hours=12),
            y1=100,
            fillcolor="LightSalmon",
            opacity=0.2,
            layer="below",
            line_width=0,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Leyenda explicativa
        st.markdown(
            """
        **Interpretación del gráfico:**
        - 🔵 Puntos azules: Registros normales (gap ≤ 30s)
        - 🔴 Puntos rojos: Posible recarga real (gap > 30s + cambio >5%)
        - 📏 Tamaño: Duración del registro (mayor tamaño = gap más largo)
        """
        )
    else:
        st.warning("No hay datos disponibles para este turno")


if __name__ == "__main__":
    main()
