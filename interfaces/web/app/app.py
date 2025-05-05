import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import sys

# Añadir directorios necesarios al path
project_root = Path(__file__).resolve().parents[2]  # Ajustar según necesidad
sys.path.append(str(project_root))

from .etl_adapter import SensorETLAdapter

# Configuración de la página
st.set_page_config(
    page_title="Monitor de Sensores",
    page_icon="🚚",
    layout="wide"
)

def main():
    st.title("📈 Monitor de Datos de Sensores en Tiempo Real")
    
    # Sidebar para controles
    with st.sidebar:
        st.header("Configuración")
        truck_id = st.text_input("ID del Camión", "T-210")
        update_btn = st.button("Actualizar Datos")
    
    # Sección principal
    if update_btn:
        try:
            etl = SensorETLAdapter(truck_id)
            df_clean, metrics = etl.run_etl()
            
            # Convertir a Pandas para visualización
            df_display = df_clean.to_pandas()
            df_display['TimeStamp'] = pd.to_datetime(df_display['TimeStamp'])
            
            # Mostrar métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Registros Iniciales", metrics['initial_records'])
            with col2:
                st.metric("Registros Limpios", metrics['cleaned_records'])
            with col3:
                st.metric("Datos Limpios", f"{metrics['clean_percentage']:.2f}%")
            
            # Gráfico de combustible
            st.subheader("Nivel de Combustible")
            fig = px.line(
                df_display,
                x='TimeStamp',
                y='FuelLevel',
                title=f"Tendencia de Combustible - {truck_id}",
                labels={'FuelLevel': 'Nivel (%)', 'TimeStamp': 'Hora'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mapa de ubicaciones
            st.subheader("Ubicaciones del Camión")
            st.map(df_display[['Latitude', 'Longitude']].dropna())
            
            # Datos crudos
            with st.expander("Ver Datos Detallados"):
                st.dataframe(df_display[['TimeStamp', 'Equipment', 'FuelLevel']])
                
        except Exception as e:
            st.error(f"Error al cargar datos: {str(e)}")

if __name__ == "__main__":
    main()