import streamlit as st
import polars as pl
from analytics.EDA.sensor.sensor_data_eda import SensorDataEDA
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from model.predictive.mlflow_config import TRUCK_IDS


# load data from database
@st.cache_resource
def get_analyzer(truck_id: str):
    analyzer = SensorDataEDA(truck_id=truck_id)
    analyzer.run()
    return analyzer


def load_data(truck_id: str):
    """Load data according to the current mode"""
    with st.spinner("Cargando datos (modo desarrollo)..."):
        analyzer = get_analyzer(truck_id)
        df = analyzer.get_dataframe()
        stats = analyzer.get_statistics()
    return analyzer, df, stats


def show():

    with st.sidebar:
        st.header("🚚 Elige un camion para analizar")
        truck_id = st.selectbox(
            "Selecciona el ID del Camión:",
            options=TRUCK_IDS,
            index=0,
        )

    st.title(f"📊 Análisis Combinado de sensor del camion {truck_id}")

    # Load data according to the mode
    analyzer, df, stats = load_data(truck_id)

    # Configuración común
    col_config = st.container()

    with col_config:
        st.header("Analisis de datos de sensores")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📈 Serie Temporal",
                "📊 Histograma",
                "📦 Box Plot",
                "🔗 Matriz Correlación",
                "⛽ Análisis Recargas",
            ]
        )
        # tab time series
        with tab1:
            st.subheader("📈 Análisis de Serie Temporal")
            min_date = df.select(pl.col("TimeStamp").dt.date()).min().item()
            max_date = df.select(pl.col("TimeStamp").dt.date()).max().item()

            # Seleccionar variable a analizar
            grouping_option = st.selectbox(
                "Seleccionar variable a analizar",
                [
                    "nivel de combustible",
                    "delta de combustible",
                    "velocidad",
                    "aceleración",
                ],
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
                    "col": "SpeedAvg",
                    "label": "Velocidad Promedio (km/h)",
                    "color": "#2ca02c",
                },
                "aceleración": {
                    "col": "Acceleration",
                    "label": "Aceleración (m/s²)",
                    "color": "#d62728",
                },
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

        with tab4:
            st.subheader("🔗 Matriz de Correlación")

            try:
                corr_data = analyzer.get_correlation_matrix()

                # Mostrar advertencia si hay columnas excluidas
                if corr_data.get("excluded_columns"):
                    with st.expander("⚠️ Columnas excluidas del análisis"):
                        for exc in corr_data["excluded_columns"]:
                            st.warning(f"**{exc['column']}**: {exc['reason']}")

                # Métricas principales
                st.markdown("### 📊 Correlaciones más fuertes")

                # Verificar que hay suficientes pares
                if len(corr_data["pairs"]) == 0:
                    st.warning("No hay correlaciones disponibles para mostrar")
                else:
                    # Mostrar top 3 (o menos si no hay suficientes)
                    num_pairs = min(3, len(corr_data["pairs"]))
                    cols = st.columns(num_pairs)

                    for i in range(num_pairs):
                        pair = corr_data["pairs"][i]
                        with cols[i]:
                            st.metric(
                                f"{pair['var1']} ↔ {pair['var2']}",
                                f"{pair['correlation']:.3f}",
                                delta=None,
                            )

                    # Heatmap de correlación
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr_data["matrix"],
                            x=corr_data["columns"],
                            y=corr_data["columns"],
                            colorscale="RdBu",
                            zmid=0,
                            zmin=-1,
                            zmax=1,
                            text=[
                                [f"{val:.2f}" for val in row]
                                for row in corr_data["matrix"]
                            ],
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            colorbar=dict(title="Correlación"),
                        )
                    )

                    fig.update_layout(
                        title="Matriz de Correlación de Variables Operacionales",
                        height=600,
                        template="plotly_white",
                        xaxis_title="",
                        yaxis_title="",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de pares ordenados
                    with st.expander("📋 Ver todas las correlaciones ordenadas"):
                        if corr_data["pairs"]:
                            pairs_df = pl.DataFrame(corr_data["pairs"])
                            st.dataframe(
                                pairs_df.to_pandas().style.background_gradient(
                                    subset=["correlation"],
                                    cmap="RdBu_r",
                                    vmin=-1,
                                    vmax=1,
                                ),
                                use_container_width=True,
                            )
                        else:
                            st.info("No hay pares de correlación disponibles")

            except Exception as e:
                st.error(f"Error al calcular correlaciones: {str(e)}")
                st.exception(e)

        with tab5:
            st.subheader("⛽ Análisis de Eventos de Recarga")

            try:
                refuel_data = analyzer.analyze_refuel_events()

                if refuel_data.get("valid_refuels", 0) == 0:
                    st.warning("⚠️ No se encontraron eventos de recarga en los datos")
                else:
                    # KPIs principales
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Recargas",
                            f"{refuel_data['valid_refuels']}",
                            help="Eventos de recarga detectados (DeltaFuel > 50L)",
                        )
                    with col2:
                        st.metric(
                            "Volumen Total",
                            f"{refuel_data['total_volume_liters']:,.0f} L",
                        )
                    with col3:
                        st.metric(
                            "Promedio por Recarga",
                            f"{refuel_data['avg_volume_liters']:.1f} L",
                        )

                    # Gráfico de distribución de volúmenes de recarga
                    col4, col5 = st.columns([2, 1])
                    with col4:
                        st.markdown("### 📊 Distribución de Volúmenes de Recarga")
                        refuel_df = refuel_data["refuel_events_df"]

                        fig_hist = px.histogram(
                            refuel_df.to_pandas(),
                            x="DeltaFuel",
                            nbins=30,
                            title="Histograma de Volúmenes de Recarga",
                            labels={
                                "DeltaFuel": "Volumen (Litros)",
                                "count": "Frecuencia",
                            },
                            color_discrete_sequence=["#2ecc71"],
                        )
                        fig_hist.add_vline(
                            x=refuel_data["avg_volume_liters"],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Media: {refuel_data['avg_volume_liters']:.1f}L",
                        )
                        fig_hist.update_layout(height=400, template="plotly_white")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Gráficos comparativos
                    with col5:
                        st.markdown("### 🔄 Recargas por Turno")
                        shift_df = pl.DataFrame(refuel_data["refuels_by_shift"])
                        fig_shift = px.bar(
                            shift_df.to_pandas(),
                            x="Shift",
                            y="count",
                            text="count",
                            title="Cantidad de Recargas por Turno",
                            color="Shift",
                            color_discrete_map={"D": "#f39c12", "N": "#3498db"},
                        )
                        fig_shift.update_traces(textposition="outside")
                        fig_shift.update_layout(
                            height=400, showlegend=False, template="plotly_white"
                        )
                        st.plotly_chart(fig_shift, use_container_width=True)

                    # Serie temporal de recargas
                    st.markdown("### 📅 Serie Temporal de Recargas")
                    fig_timeline = px.scatter(
                        refuel_df.to_pandas(),
                        x="TimeStamp",
                        y="DeltaFuel",
                        color="ValidFuel",
                        size="DeltaFuel",
                        title="Eventos de Recarga en el Tiempo",
                        labels={
                            "TimeStamp": "Fecha y Hora",
                            "DeltaFuel": "Volumen (Litros)",
                            "ValidFuel": "Válida",
                        },
                        color_discrete_map={0: "#e74c3c", 1: "#27ae60"},
                        hover_data=["Shift", "BeforeAvg", "AfterAvg"],
                    )
                    fig_timeline.update_layout(height=500, template="plotly_white")
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Tabla detallada
                    with st.expander("📋 Ver tabla detallada de recargas"):
                        display_df = refuel_df.with_columns(
                            [
                                pl.col("TimeStamp")
                                .dt.strftime("%Y-%m-%d %H:%M")
                                .alias("Fecha"),
                                pl.col("Shift").alias("Turno"),
                                pl.col("DeltaFuel").round(2).alias("Volumen (L)"),
                                pl.col("BeforeAvg").round(2).alias("Antes (L)"),
                                pl.col("AfterAvg").round(2).alias("Después (L)"),
                                pl.col("ValidFuel").cast(pl.Utf8).alias("Válida"),
                            ]
                        ).select(
                            [
                                "Fecha",
                                "Turno",
                                "Volumen (L)",
                                "Antes (L)",
                                "Después (L)",
                                "Válida",
                            ]
                        )

                        st.dataframe(
                            display_df.to_pandas(), use_container_width=True, height=400
                        )

            except Exception as e:
                st.error(f"Error al analizar recargas: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    show()
