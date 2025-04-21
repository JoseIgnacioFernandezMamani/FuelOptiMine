import streamlit as st
import plotly.express as px

st.title("📈 Modelo Lineal")
st.markdown("Análisis de consumo base usando regresión lineal")

# Datos de ejemplo
data = {
    'Tonelaje': [100, 200, 300, 400, 500],
    'Consumo Real': [120, 135, 128, 140, 132],
    'Predicción': [115, 130, 140, 145, 150]
}

# Widgets de control
with st.sidebar:
    st.header("Parámetros del Modelo")
    intercepto = st.slider("Intercepto", 50, 200, 100)
    coef_tonelaje = st.slider("Coef. Tonelaje", 0.1, 0.5, 0.3)

# Gráfico interactivo
df = pd.DataFrame(data)
df['Predicción Ajustada'] = intercepto + coef_tonelaje * df['Tonelaje']

fig = px.line(df, x='Tonelaje', y=['Consumo Real', 'Predicción Ajustada'],
              title="Comparación Predicción vs Real")
st.plotly_chart(fig, use_container_width=True)