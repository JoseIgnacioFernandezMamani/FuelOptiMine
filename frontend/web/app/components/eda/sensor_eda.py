import streamlit as st
import polars as pl
from analytics.EDA.sensor.sensor_data_eda import SensorDataEDA
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# Mode configuration (True for development, False for production)
DEV_MODE = True  # Change to False for production deployment


# load data from database
@st.cache_resource
def get_analyzer():
    analyzer = SensorDataEDA(truck_id="T-235")
    analyzer.run()
    return analyzer


def load_data():
    """Load data according to the current mode"""
    if DEV_MODE:
        with st.spinner("Cargando datos (modo desarrollo)..."):
            analyzer = get_analyzer()
            df = analyzer.get_dataframe()
            stats = analyzer.get_statistics()
        return analyzer, df, stats
    else:
        if "analyzer" not in st.session_state:
            with st.spinner("Cargando datos (modo producción)..."):
                st.session_state.analyzer = get_analyzer()
                st.session_state.df = st.session_state.analyzer.get_dataframe()
                st.session_state.stats = st.session_state.analyzer.get_statistics()
        return (st.session_state.analyzer, st.session_state.df, st.session_state.stats)


def show():
    st.title("📊 Análisis Combinado de Combustible")

    if DEV_MODE:
        st.warning("MODO DESARROLLO ACTIVO - Los datos se recargan en cada cambio")
    else:
        st.info("MODO PRODUCCIÓN - Los datos se mantienen en memoria cache")

    # Load data according to the mode
    analyzer, df, stats = load_data()

    # Configuración común
    col_config = st.container()

    with col_config:
        st.header("Analisis de datos de sensores")
        tab1, tab2, tab3 = st.tabs(
            ["📈 Serie Temporal", "📊 Histograma", "📦 Box Plot"]
        )
        # tab time series
        with tab1:
            st.subheader("📈 Análisis de Serie Temporal")
            min_date = df.select(pl.col("TimeStamp").dt.date()).min().item()
            max_date = df.select(pl.col("TimeStamp").dt.date()).max().item()

            # Seleccionar variable a analizar
            grouping_option = st.selectbox(
                "Seleccionar variable a analizar",
                ["nivel de combustible", "delta de combustible", "velocidad", "RPM"],
                index=0,  # Default is "nivel de combustible"
            )

            # Select analysis period
            period_option = st.selectbox(
                "Seleccionar período de análisis",
                ["Dia", "Semana", "Mes", "Trimestre"],
                index=0,  # default is "Dia"
            )

            # Map variables to columns
            variable_mapping = {
                "nivel de combustible": {
                    "col": "FuelLevelLiters",
                    "label": "Litros",
                    "color": "#1f77b4",
                },
                "delta de combustible": {
                    "col": "DeltaFuel",
                    "label": "Delta Combustible (L)",
                    "color": "#ff7f0e",
                },
                "velocidad": {
                    "col": "Speed",
                    "label": "Velocidad (km/h)",
                    "color": "#2ca02c",
                },
                "RPM": {"col": "RPM", "label": "RPM", "color": "#d62728"},
            }

            selected_var = variable_mapping[grouping_option]

            if period_option == "Dia":
                selected_date = st.date_input(
                    "Seleccionar fecha para análisis temporal",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                )
                filtered_data = df.filter(
                    pl.col("TimeStamp").dt.date() == selected_date
                )

                if not filtered_data.is_empty():
                    x_axis = filtered_data["TimeStamp"].to_numpy()
                    y_axis = filtered_data[selected_var["col"]].to_numpy()

                    fig_temporal = go.Figure()
                    fig_temporal.add_trace(
                        go.Scattergl(
                            x=x_axis,
                            y=y_axis,
                            mode="lines",
                            name=grouping_option.title(),
                            line=dict(color=selected_var["color"], width=2),
                        )
                    )
                    fig_temporal.update_layout(
                        title=f"{grouping_option.title()} - {selected_date}",
                        xaxis_title="Hora del Dia",
                        yaxis_title=selected_var["label"],
                        hovermode="x unified",
                        template="plotly_white",
                        height=500,
                    )
                    st.plotly_chart(fig_temporal, use_container_width=True)
                else:
                    st.warning(f"No hay datos para el {selected_date}")

            else:
                # Análisis con resampling
                if not df.is_empty():
                    # Preparar datos para resampling
                    df_sorted = df.sort("TimeStamp")

                    # Configurar el período de resampling
                    if period_option == "Semana":
                        resample_period = "1w"
                        title_suffix = "por Semana"
                        x_title = "Semanas"
                    elif period_option == "Mes":
                        resample_period = "1mo"
                        title_suffix = "por Mes"
                        x_title = "Meses"
                    elif period_option == "Trimestre":
                        resample_period = "1q"
                        title_suffix = "por Trimestre"
                        x_title = "Trimestres"

                    # Realizar resampling con Polars
                    resampled_data = (
                        df_sorted.group_by_dynamic("TimeStamp", every=resample_period)
                        .agg(
                            [
                                pl.col(selected_var["col"]).mean().alias("mean_value"),
                                pl.col(selected_var["col"]).min().alias("min_value"),
                                pl.col(selected_var["col"]).max().alias("max_value"),
                                pl.col(selected_var["col"]).count().alias("count"),
                            ]
                        )
                        .filter(pl.col("count") > 0)  # Filtrar períodos sin datos
                        .sort("TimeStamp")
                    )

                    if not resampled_data.is_empty():
                        x_axis = resampled_data["TimeStamp"].to_numpy()
                        y_mean = resampled_data["mean_value"].to_numpy()
                        y_min = resampled_data["min_value"].to_numpy()
                        y_max = resampled_data["max_value"].to_numpy()

                        fig_temporal = go.Figure()

                        # Agregar línea de promedio
                        fig_temporal.add_trace(
                            go.Scattergl(
                                x=x_axis,
                                y=y_mean,
                                mode="lines+markers",
                                name=f"Promedio {grouping_option.title()}",
                                line=dict(color=selected_var["color"], width=2),
                                marker=dict(size=4),
                            )
                        )

                        # Agregar área de rango (min-max)
                        fig_temporal.add_trace(
                            go.Scatter(
                                x=x_axis,
                                y=y_max,
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hovertemplate="Máximo: %{y}<extra></extra>",
                            )
                        )

                        fig_temporal.add_trace(
                            go.Scatter(
                                x=x_axis,
                                y=y_min,
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor=f'rgba({int(selected_var["color"][1:3], 16)}, {int(selected_var["color"][3:5], 16)}, {int(selected_var["color"][5:7], 16)}, 0.2)',
                                name="Rango Min-Max",
                                hovertemplate="Mínimo: %{y}<extra></extra>",
                            )
                        )

                        fig_temporal.update_layout(
                            title=f"Promedio de {grouping_option.title()} {title_suffix}",
                            xaxis_title=x_title,
                            yaxis_title=selected_var["label"],
                            hovermode="x unified",
                            template="plotly_white",
                            height=500,
                            showlegend=True,
                        )

                        st.plotly_chart(fig_temporal, use_container_width=True)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Períodos analizados", len(resampled_data))
                        with col2:
                            st.metric("Promedio general", f"{y_mean.mean():.2f}")
                        with col3:
                            st.metric("Segundo valor mínimo", f"{y_min.min():.2f}")
                        with col4:
                            st.metric("Valor máximo", f"{y_max.max():.2f}")

                    else:
                        st.warning(
                            f"No hay datos suficientes para el análisis {title_suffix.lower()}"
                        )
                else:
                    st.warning("No hay datos disponibles para el análisis")
        # Tab histogram
        with tab2:
            st.subheader("📊 Histogramas")

            # Initialize states
            if "manual_bins" not in st.session_state:
                st.session_state.manual_bins = 30
            if "use_manual" not in st.session_state:
                st.session_state.use_manual = False

            with st.container():

                calc_auto = st.checkbox("Cálculo automático", key="disable")

                col1, col2 = st.columns(2)
                with col1:
                    selected_col = st.selectbox(
                        "Seleccione la variable:",
                        options=list(stats.keys()),
                        index=1,
                    )

                with col2:
                    if calc_auto:
                        method = st.selectbox(
                            "Método de cálculo:",
                            ["auto", "fd", "scott", "sturges", "sqrt", "choice"],
                        )
                    else:
                        method = "sqrt"
                        st.write("")
                        st.info("Modo manual - Control directo")

                # Calculate dynamic range for the slider
                col_stats = stats[selected_col]
                max_val = int(col_stats.get("max", 1)) or 1
                min_val = int(col_stats.get("min", 0)) or 0
                non_null_count = col_stats.get("non_null_count", 1000)
                data_range = max_val - min_val

                if data_range > 0:
                    max_bins_sturges = int(np.ceil(np.log2(non_null_count) + 1))
                    max_bins_sqrt = int(np.sqrt(non_null_count))
                    min_slider = 5
                    max_slider = min(
                        200, max(50, max_bins_sturges * 3, max_bins_sqrt * 2)
                    )
                else:
                    min_slider = 5
                    max_slider = 50

                if calc_auto:
                    auto_bins = analyzer.calculate_bins(
                        column=selected_col, method=method
                    )
                    final_bins = auto_bins
                    st.session_state.manual_bins = 0
                    st.session_state.use_manual = False
                else:
                    manual_bins = st.slider(
                        "Número de bins:",
                        min_value=min_slider,
                        max_value=max_slider,
                        value=(
                            st.session_state.manual_bins
                            if st.session_state.manual_bins > 0
                            else int(np.sqrt(non_null_count))
                        ),
                        help=f"Rango de datos: {min_val} - {max_val}. Control manual directo",
                    )
                    st.session_state.manual_bins = manual_bins
                    final_bins = manual_bins

            with st.container():
                st.markdown("### 📈 Visualización del Histograma")

                col_chart, col_metrics = st.columns([3, 1])
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

                with col_metrics:
                    st.metric("Bins usados", final_bins)
                    st.metric("Método", method)
                    st.metric("Datos", f"{non_null_count:,}")
                    st.metric("Rango máx. slider", max_slider)

            with st.container():
                with st.expander("📋 Ver estadísticas de la variable"):
                    st.json(col_stats)

        # tab boxplot
        with tab3:
            st.subheader("📦 Box Plot: Distribución de Nivel de Combustible")

            # Select grouping
            grouping_option = st.selectbox(
                "Seleccionar agrupación temporal:",
                ["Month", "Quarter", "All Data"],
            )

            if grouping_option == "Month":
                col1, col2 = st.columns(2)
                with col1:
                    selected_month = st.selectbox(
                        "Seleccionar mes:",
                        range(1, 13),
                        format_func=lambda x: [
                            "Enero",
                            "Febrero",
                            "Marzo",
                            "Abril",
                            "Mayo",
                            "Junio",
                            "Julio",
                            "Agosto",
                            "Septiembre",
                            "Octubre",
                            "Noviembre",
                            "Diciembre",
                        ][x - 1],
                        index=max_date.month - 1,
                    )
                with col2:
                    available_years = sorted(
                        df["TimeStamp"].dt.year().unique().to_list()
                    )
                    selected_year = st.selectbox(
                        "Seleccionar año:",
                        available_years,
                        index=(
                            available_years.index(max_date.year)
                            if max_date.year in available_years
                            else 0
                        ),
                    )

                filtered_df = df.filter(
                    (df["TimeStamp"].dt.month() == selected_month)
                    & (df["TimeStamp"].dt.year() == selected_year)
                )
                period_format = "%Y-%m-%d"
                period_label = "Dia"

            elif grouping_option == "Quarter":
                col1, col2 = st.columns(2)
                with col1:
                    selected_quarter = st.selectbox(
                        "Seleccionar trimestre:",
                        [1, 2, 3, 4],
                        format_func=lambda x: f"Q{x} ({['Ene-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec'][x-1]})",
                        index=((max_date.month - 1) // 3),
                    )
                with col2:
                    available_years = sorted(
                        df["TimeStamp"].dt.year().unique().to_list()
                    )
                    selected_year = st.selectbox(
                        "Seleccionar año:",
                        available_years,
                        index=(
                            available_years.index(max_date.year)
                            if max_date.year in available_years
                            else 0
                        ),
                    )

                quarter_months = {
                    1: [1, 2, 3],
                    2: [4, 5, 6],
                    3: [7, 8, 9],
                    4: [10, 11, 12],
                }

                filtered_df = df.filter(
                    (df["TimeStamp"].dt.month().is_in(quarter_months[selected_quarter]))
                    & (df["TimeStamp"].dt.year() == selected_year)
                )
                period_format = "%Y-%m"
                period_label = "Mes"

            else:
                st.info("📊 Mostrando todos los datos disponibles")
                filtered_df = df
                period_format = "%Y-%m"
                period_label = "Mes"

            if filtered_df.is_empty():
                st.warning("⚠️ No hay datos disponibles para el período seleccionado.")
            else:
                plot_df = filtered_df.with_columns(
                    pl.col("TimeStamp").dt.strftime(period_format).alias("Period")
                )

                if plot_df["FuelLevelLiters"].null_count() == len(plot_df):
                    st.warning("⚠️ No hay datos de combustible disponibles.")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total de registros", f"{len(plot_df):,}")
                    with col2:
                        st.metric("📅 Períodos únicos", len(plot_df["Period"].unique()))
                    with col3:
                        fuel_mean = plot_df["FuelLevelLiters"].mean()
                        st.metric(
                            "⛽ Promedio combustible",
                            f"{fuel_mean:.1f}L" if fuel_mean else "N/A",
                        )

                    fig = px.box(
                        plot_df,
                        x="Period",
                        y="FuelLevelLiters",
                        points="outliers",
                        title=f"Distribución de Nivel de Combustible por {period_label}",
                        color_discrete_sequence=["#636EFA"],
                        labels={
                            "Period": period_label,
                            "FuelLevelLiters": "Nivel de Combustible (Litros)",
                        },
                    )

                    fig.update_layout(
                        xaxis_title=period_label,
                        yaxis_title="Nivel de Combustible (Litros)",
                        template="plotly_white",
                        height=500,
                        hovermode="x unified",
                        xaxis_tickangle=-45,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📊 Estadísticas por período"):
                        stats_df = (
                            plot_df.group_by("Period")
                            .agg(
                                pl.col("FuelLevelLiters").count().alias("Registros"),
                                pl.col("FuelLevelLiters")
                                .mean()
                                .round(2)
                                .alias("Promedio"),
                                pl.col("FuelLevelLiters")
                                .std()
                                .round(2)
                                .alias("Desv_Std"),
                                pl.col("FuelLevelLiters").min().alias("Mínimo"),
                                pl.col("FuelLevelLiters").max().alias("Máximo"),
                                pl.col("FuelLevelLiters")
                                .median()
                                .round(2)
                                .alias("Mediana"),
                            )
                            .sort("Period")
                        )
                        st.dataframe(stats_df.to_pandas(), use_container_width=True)


if __name__ == "__main__":
    show()
