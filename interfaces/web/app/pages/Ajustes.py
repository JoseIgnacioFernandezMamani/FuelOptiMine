import streamlit as st

st.title("⚙️ Configuraciones")
st.markdown("Personalización del sistema")

# Configuración de conexiones
with st.expander("Conexión a Bases de Datos"):
    st.text_input("Servidor SQL", "sql.msc.local")
    st.text_input("Usuario", type="password")
    st.text_input("Contraseña", type="password")

# Configuración de modelos
with st.expander("Parámetros Avanzados"):
    st.checkbox("Usar GPU para inferencia", value=False)
    st.slider("Frecuencia de actualización (min)", 1, 60, 15)

if st.button("Guardar Cambios"):
    st.success("Configuraciones actualizadas correctamente")