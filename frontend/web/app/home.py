import streamlit as st

# Configuración inicial
st.set_page_config(
    page_title="Fuel Analytics", page_icon="./images/logo.png", layout="wide"
)

# Sistema de autenticación simplificado
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def login():
    st.header("Autenticación")
    with st.form("login_form"):
        user = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")

        if st.form_submit_button("Ingresar"):
            if user == "admin" and password == "admin":
                st.session_state.authenticated = True
                st.rerun()


def logout():
    st.session_state.authenticated = False
    st.query_params.clear()
    st.markdown(
        '<meta http-equiv="refresh" content="0; url=/" />', unsafe_allow_html=True
    )
    st.stop()


home = st.Page("components/home/home.py", title="Inicio", icon="🏠")
sensor_eda = st.Page("components/eda/sensor_eda.py", title="EDA de Sensores", icon="🚥")
cycle_eda = st.Page("components/eda/cycle_eda.py", title="EDA de Ciclos", icon="🔁")
supply_eda = st.Page(
    "components/eda/supply_eda.py", title="EDA de Combustible", icon="⛽"
)
time_model_eda = st.Page(
    "components/eda/time_model_eda.py", title="EDA de Modelo de Tiempo", icon="📅"
)
model = st.Page("components/models/model.py", title="Modelo Predictivo", icon="📈")
config = st.Page("components/config/config.py", title="Configuración", icon="⚙️")

# Navegación dinámica
if not st.session_state.authenticated:
    login()
else:
    # Menú principal
    st.logo("./images/logo.png", icon_image="./images/logo.png")
    nav_sections = {
        "Inicio": [home],
        "Analisis": [cycle_eda, sensor_eda, supply_eda, time_model_eda],
        "Modelo": [model],
        "Configuración": [config],
        "Cerrar Sesión": [st.Page(logout, title="Cerrar Sesión", icon="🚪")],
    }

    pg = st.navigation(nav_sections)
    pg.run()
