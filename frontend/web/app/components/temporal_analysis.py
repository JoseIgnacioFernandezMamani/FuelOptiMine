import streamlit as st
from analytics.EDA.sensor.sensor_data_eda import DistributionAnalyzer
import plotly.express as px


def show():

    st.title("📈 Análisis Temporal de Combustible")

    if "analyzer" not in st.session_state:
        st.session_state.analyzer = DistributionAnalyzer()
        st.session_state.df = st.session_state.analyzer.load_sensor_data()
        st.session_state.stats = st.session_state.analyzer.generate_statistics()

    with st.sidebar:
        st.header("Configuración del análisis")
        selected_col = st.selectbox(
            "Seleccione columna:",
            options=st.session_state.analyzer._stats_cache.keys(),
            index=1,
        )

        method = st.selectbox(
            "Método de cálculo de bins:", ["auto", "fd", "scott", "sturges", "sqrt"]
        )
    # Contenedor principal
    with st.container():
        # Sección de visualización
        col1, col2 = st.columns([3, 1])

        with col1:
            # Calcular bins con método seleccionado
            bins = st.session_state.analyzer.calculate_optimal_bins(
                selected_col, method=method
            )

            # Crear histograma
            fig = px.histogram(
                st.session_state.analyzer.sensor_df.to_pandas(),
                x=selected_col,
                nbins=bins,
                title=f"Distribución de {selected_col}",
                labels={selected_col: selected_col},
                opacity=0.7,
                color_discrete_sequence=["#FF7F0E"],
            )

            fig.update_layout(
                xaxis_title=selected_col, yaxis_title="Frecuencia", bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Mostrar estadísticas clave
            stats = st.session_state.analyzer._stats_cache[selected_col]
            st.metric("Bins calculados", bins)
            st.metric("Media", f"{stats['mean']:.2f}")
            st.metric("Mediana", f"{stats['median']:.2f}")
            st.metric("Desviación Estándar", f"{stats['std_dev']:.2f}")

    # Sección de estadísticas completas
    with st.expander("Ver estadísticas completas"):
        st.json(st.session_state.analyzer._stats_cache[selected_col])
