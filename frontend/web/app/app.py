import streamlit as st
import importlib

# Configuración inicial
st.set_page_config(page_title="FuelOptiMine Analytics", page_icon="⛽", layout="wide")

# Mapeo de páginas (nombre visible: nombre_archivo)
PAGES = {"🏠 Inicio": "home", "📈 Temporal": "temporal_analysis"}

# Navegación en sidebar
selected = st.sidebar.radio("Menú", list(PAGES.keys()))

# Carga dinámica de la página seleccionada
module = importlib.import_module(f"pages.{PAGES[selected]}")
module.show()
