import streamlit as st


def show():
    st.title("Bienvenido a FuelOptiMine")
    st.write("Seleccione un análisis en el menú lateral")

    # Carga de datos centralizada (ejemplo)
    if "data" not in st.session_state:
        from analitycs.EDA.sensor.distribution_analyzer import FuelAnalysisOptimized

        analyzer = FuelAnalysisOptimized()
        st.session_state.data = analyzer.load_sensor_data()

    st.write("Datos cargados:", st.session_state.data.shape)
