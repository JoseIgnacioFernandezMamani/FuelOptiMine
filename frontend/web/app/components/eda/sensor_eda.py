import streamlit as st
from analitycs.EDA.sensor.sensor_data_eda import SensorDataEDA
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_elements import elements, dashboard, mui, html


@st.cache_resource
def get_analyzer():
    analyzer = SensorDataEDA(truck_id="T-212")
    analyzer.run()
    return analyzer


def show():
    st.title("📊 Análisis Combinado de Combustible")

    # Inicialización de estados
    if "analyzer" not in st.session_state:
        with st.spinner("Cargando datos..."):
            st.session_state.analyzer = get_analyzer()
            st.session_state.df = st.session_state.analyzer.get_dataframe()
            st.session_state.stats = st.session_state.analyzer.get_statistics()
        st.success("✅ Datos cargados correctamente.")

    # Configuración común
    # df = st.session_state.df.to_pandas()  # Convertir a Pandas para Plotly
    df = st.session_state.df

    col_config = st.container()

    with col_config:
        st.header("Configuración de Visualizaciones")
        tab1, tab2 = st.tabs(["📈 Serie Temporal", "📊 Histograma"])

        # Pestaña de Serie Temporal
        with tab1:

            min_date = df["TimeStamp"].dt.date().min()
            max_date = df["TimeStamp"].dt.date().max()

            selected_date = st.date_input(
                "Seleccionar fecha para análisis temporal",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )

            filtered_data = df.filter(df["TimeStamp"].dt.date() == selected_date)

            # Crear gráfico temporal
            if not filtered_data.is_empty():
                fig_temporal = go.Figure()
                fig_temporal.add_trace(
                    go.Scattergl(
                        x=filtered_data["TimeStamp"].to_numpy(),
                        y=filtered_data["FuelLevelLiters"].to_numpy(),
                        mode="lines",
                        name="Nivel de Combustible",
                        line=dict(color="#1f77b4", width=1),
                    )
                )

                fig_temporal.update_layout(
                    title=f"Variación de Combustible - {selected_date}",
                    xaxis_title="Hora del día",
                    yaxis_title="Litros",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500,
                )

                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                st.warning(f"No hay datos para el {selected_date}")

        # Pestaña de Histograma (tu código original)
        with tab2:
            if "manual_bins" not in st.session_state:
                st.session_state.manual_bins = None
            if "use_manual" not in st.session_state:
                st.session_state.use_manual = False

            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox(
                    "Seleccione columna:",
                    options=st.session_state.analyzer._stats_cache.keys(),
                    index=1,
                )
            with col2:
                method = st.selectbox(
                    "Método de cálculo:", ["auto", "fd", "scott", "sturges", "sqrt"]
                )

            stats = st.session_state.stats[selected_col]
            max_val = int(stats.get("max", 1)) or 1
            min_val = int(stats.get("min", 0)) or 0

            auto_bins = st.session_state.analyzer.calculate_bins(
                column=selected_col, method=method
            )

            with st.container():
                col_slider, _ = st.columns([3, 1])
                with col_slider:
                    manual_bins = st.slider(
                        "Bins manuales (0 para auto)",
                        min_value=0,
                        max_value=min(100, max_val),
                        value=st.session_state.manual_bins or 0,
                        help=f"Rango de datos: {min_val} - {max_val}",
                    )

                    if manual_bins != st.session_state.manual_bins:
                        st.session_state.manual_bins = manual_bins
                        st.session_state.use_manual = manual_bins > 0

                    current_selection = f"{selected_col}-{method}"
                    if "last_selection" not in st.session_state:
                        st.session_state.last_selection = current_selection
                    if st.session_state.last_selection != current_selection:
                        st.session_state.manual_bins = 0
                        st.session_state.use_manual = False
                        st.session_state.last_selection = current_selection

                final_bins = (
                    st.session_state.manual_bins
                    if st.session_state.use_manual
                    else auto_bins
                )

                col_chart, col_stats = st.columns([3, 1])
                with col_chart:
                    fig = px.histogram(
                        df,
                        x=selected_col,
                        nbins=final_bins,
                        title=f"Distribución de {selected_col}",
                        labels={selected_col: "Valor"},
                        color_discrete_sequence=["#FF4B4B"],
                        opacity=0.8,
                    )
                    fig.update_layout(
                        bargap=0.1, xaxis_title=selected_col, yaxis_title="Frecuencia"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    st.metric("Bins usados", final_bins)
                    st.metric("Mínimo", f"{stats['min']:.2f}")
                    st.metric("Máximo", f"{stats['max']:.2f}")
                    st.metric("Media", f"{stats['mean']:.2f}")

                with st.expander("📊 Estadísticas completas"):
                    st.json(st.session_state.stats[selected_col])


if __name__ == "__main__" or __name__ == "__streamlit__":
    show()
