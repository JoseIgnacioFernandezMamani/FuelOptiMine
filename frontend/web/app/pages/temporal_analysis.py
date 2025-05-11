import streamlit as st
from components.charts.histogram import plot_histogram
from analitycs.EDA.sensor.distribution_analyzer import FuelAnalysisOptimized


def show():
    st.title("Análisis Temporal de Combustible")

    # Cargar datos una sola vez
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = FuelAnalysisOptimized()
        st.session_state.df = st.session_state.analyzer.load_sensor_data()

    # Usar componente reutilizable
    plot_histogram(
        st.session_state.df, "FuelLevelLiters", "Distribución de Niveles de Combustible"
    )

    # Mostrar estadísticas
    if st.button("Calcular Métricas"):
        stats = st.session_state.analyzer.generate_statistics()
        st.json(stats)
