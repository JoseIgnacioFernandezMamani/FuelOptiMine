import streamlit as st


def show():
    st.title("⛽ FuelOptiMine Analytics")
    st.markdown(
        """
    ## Bienvenido al Sistema de Análisis de Combustible
        
    **Seleccione una opción del menú lateral para comenzar:**
    - 📈 Temporal: Análisis de distribución temporal
    - 🔍 Comparativo: Comparación entre múltiples variables
    - 📊 Tendencia: Análisis de tendencias históricas
    """
    )

    # Carga de datos con progreso
    if "data_loaded" not in st.session_state:
        with st.spinner("Cargando datos iniciales..."):
            from analitycs.EDA.sensor.distribution_analyzer import DistributionAnalyzer

            analyzer = DistributionAnalyzer()
            st.session_state.data = analyzer.load_sensor_data().to_pandas()
            st.session_state.data_loaded = True

    st.success("✅ Datos cargados exitosamente")
    st.dataframe(st.session_state.data.head(), use_container_width=True)
