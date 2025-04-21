""" import streamlit as st
import shap
import matplotlib.pyplot as plt

st.title("🌳 Modelo XGBoost")
st.markdown("Predicciones avanzadas con boosting de árboles")

# Simulación de importancia de características
features = {
    'Tonelaje': 0.35,
    'Pendiente': 0.28,
    'Antigüedad': 0.17,
    'Temperatura': 0.12,
    'Humedad': 0.08
}

# Widgets
with st.sidebar:
    st.header("Hiperparámetros")
    n_estimators = st.selectbox("Número de árboles", [100, 200, 300])
    learning_rate = st.slider("Tasa de aprendizaje", 0.01, 0.3, 0.1)

# Gráfico SHAP
st.subheader("Importancia de Variables (SHAP)")
fig, ax = plt.subplots()
shap_values = list(features.values())
shap.summary_plot(shap_values, features=None, feature_names=list(features.keys()), show=False)
st.pyplot(fig) """

import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [5, 6, 7, 8]
}) 

st.write('mensaje inicial')

st  .write(df)
