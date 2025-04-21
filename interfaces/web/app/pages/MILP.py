import streamlit as st
import pandas as pd

st.title("⚙️ Optimización MILP")
st.markdown("Asignación óptima de rutas y recursos")

# Entradas de usuario
camiones = st.number_input("Número de camiones", 1, 20, 5)
tonelaje_total = st.number_input("Tonelaje objetivo (ton)", 1000, 10000, 5000)

# Simulación de resultados
if st.button("Optimizar"):
    resultados = pd.DataFrame({
        'Ruta': ['Norte', 'Sur', 'Este'],
        'Camiones Asignados': [3, 2, camiones-5],
        'Combustible Estimado (L)': [450, 320, 280]
    })
    
    st.subheader("Resultados de Optimización")
    st.dataframe(resultados.style.highlight_min(subset="Combustible Estimado (L)"))
    
    st.metric("Ahorro Estimado", "15%", "-350 L/día")