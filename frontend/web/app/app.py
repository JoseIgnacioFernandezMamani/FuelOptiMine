import streamlit as st
from pathlib import Path
import sys
import os


# CONFIGURACIÓN CRÍTICA: Encontrar y agregar directorio raíz al path
def setup_project_path():
    """Configurar el path del proyecto para permitir importaciones desde la raíz"""
    current_file = Path(__file__).resolve()

    # Buscar el directorio que contiene 'analytics'
    search_path = current_file.parent
    project_root = None

    # Subir en la jerarquía hasta encontrar el directorio con 'analytics'
    for i in range(5):  # Buscar máximo 5 niveles arriba
        if (search_path / "analytics").exists():
            project_root = search_path
            break
        search_path = search_path.parent

    if project_root is None:
        # Fallback: asumir estructura conocida
        project_root = current_file.parent.parent.parent.parent

    # Agregar al path si no está ya
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    print(f"🔧 Directorio raíz configurado: {project_root}")
    print(f"📁 Analytics existe: {(project_root / 'analytics').exists()}")

    return project_root


# Ejecutar configuración
setup_project_path()

# garbage configuration
BASE_DIR: Path = Path(__file__).resolve().parent
LOGO_PATH: Path = BASE_DIR / "images" / "logo.png"

# Configuración inicial
st.set_page_config(page_title="Fuel Analytics", page_icon=LOGO_PATH, layout="wide")

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
    st.logo(LOGO_PATH, icon_image=LOGO_PATH)
    nav_sections = {
        "Inicio": [home],
        "Analisis": [cycle_eda, sensor_eda, supply_eda, time_model_eda],
        "Modelo": [model],
        "Configuración": [config],
        "Cerrar Sesión": [st.Page(logout, title="Cerrar Sesión", icon="🚪")],
    }

    pg = st.navigation(nav_sections)
    pg.run()

#
#
