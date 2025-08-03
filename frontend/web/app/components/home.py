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
