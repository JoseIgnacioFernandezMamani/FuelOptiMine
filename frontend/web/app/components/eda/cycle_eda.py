import streamlit as st
import polars as pl
from analytics.EDA.cycle.cycle_data_eda import CycleDataEDA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from mlflow_server.config import TRUCK_IDS


@st.cache_resource
def get_analyzer(truck_id: str):
    """Cargar y cachear el analizador de datos"""
    analyzer = CycleDataEDA(truck_id=truck_id)
    analyzer.run()
    return analyzer


def load_data(truck_id: str):
    """Cargar datos según el modo y el truck_id seleccionado"""

    with st.spinner(f"Cargando datos de ciclos para {truck_id}..."):
        analyzer = get_analyzer(truck_id)
        df = analyzer.get_dataframe()
        stats = analyzer.get_statistics()
        efficiency_stats = analyzer.analyze_time_efficiency()
    return analyzer, df, stats, efficiency_stats


def show():

    with st.sidebar:
        st.header("🚚 Elige un camion para analizar")
        truck_id = st.selectbox(
            "Selecciona el ID del Camión:",
            options=TRUCK_IDS,
            index=0,
        )
    st.write("Camion seleccionado:", truck_id)

    st.title(f"🚛 Análisis de Ciclos Mineros - {truck_id}")

    analyzer, df, stats, efficiency_stats = load_data(truck_id)

    # main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Resumen General",
            "⏱️ Eficiencia por Stage",
            "🔄 Eficiencia por Ciclo",
            "📍 Factores Stage 4 & 8",
            "📅 Patrones Temporales",
            "📈 Estadísticas Detalladas",
        ]
    )

    # TAB 1: GENERAL SUMMARY
    with tab1:
        st.subheader("📊 Resumen General de Datos")

        # Métricas principales
        col1, col2, col3 = st.columns(3)
        with col1:
            total_cycles = stats.get("CycleId", {}).get("total_cycles", 0)
            st.metric("🔄 Total Ciclos", f"{total_cycles:,}")

        with col2:
            avg_duration = df["CycleDurationSeconds"].mean()
            if avg_duration:
                st.metric("⏱️ Duración Promedio", f"{abs(avg_duration):.2f} seg")

        with col3:
            avg_tonnage = df.filter(pl.col("StageSequence") == 8)[
                "MeasuredTonnage"
            ].mean()
            if avg_tonnage:
                st.metric("⚖️ Tonelaje Promedio", f"{avg_tonnage:.2f} ton")

        # temporal info
        st.markdown("### 📅 Rango Temporal")
        col1, col2, col3 = st.columns(3)

        if "TimeStampIni" in stats:
            with col1:
                st.info(
                    f"**Inicio:** {stats['TimeStampIni'].get('first_record', 'N/A')}"
                )
            with col2:
                st.info(f"**Fin:** {stats['TimeStampIni'].get('last_record', 'N/A')}")
            with col3:
                duration_days = stats["TimeStampIni"].get("total_duration_days", 0)
                st.info(f"**Duración:** {duration_days:.1f} días")

        # Stage Distribution
        st.markdown("### 🔢 Distribución por Stage")
        stage_counts = (
            df.group_by("StageSequence")
            .agg(pl.len().alias("count"))
            .sort("StageSequence")
        )

        fig_stages = px.bar(
            stage_counts.to_pandas(),
            x="StageSequence",
            y="count",
            title="Cantidad de Registros por Stage",
            labels={"StageSequence": "Stage", "count": "Cantidad"},
            color="count",
            color_continuous_scale="Viridis",
        )
        fig_stages.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_stages, use_container_width=True)

        # Distribution by shift and hour
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌓 Distribución por Turno")
            shift_counts = df.group_by("Shift").agg(pl.len().alias("count"))
            fig_shift = px.pie(
                shift_counts.to_pandas(),
                values="count",
                names="Shift",
                title="Día (D) vs Noche (N)",
                color_discrete_sequence=["#FDB813", "#1f77b4"],
            )
            st.plotly_chart(fig_shift, use_container_width=True)

        with col2:
            st.markdown("### ⏰ Distribución por Hora")
            if "Hour_distribution" in stats:
                hour_data = pl.DataFrame(stats["Hour_distribution"])
                fig_hour = px.bar(
                    hour_data.to_pandas(),
                    x="Hour",
                    y="count",
                    title="Registros por Hora del Día",
                    color="count",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_hour, use_container_width=True)

    # TAB 2: EFFICIENCY BY STAGE
    with tab2:
        st.subheader("⏱️ Análisis de Eficiencia por Stage")

        eff_by_stage = pl.DataFrame(efficiency_stats["efficiency_by_stage"])

        fig_eff_stage = go.Figure()
        fig_eff_stage.add_trace(
            go.Bar(
                x=eff_by_stage["StageSequence"].to_list(),
                y=eff_by_stage["AvgEfficiency"].to_list(),
                marker_color=eff_by_stage["AvgEfficiency"].to_list(),
                marker_colorscale="RdYlGn",
                text=[f"{val:.1f}%" for val in eff_by_stage["AvgEfficiency"].to_list()],
                textposition="outside",
            )
        )
        fig_eff_stage.update_layout(
            title="Eficiencia Media por Stage (1-8)",
            xaxis_title="Stage Sequence",
            yaxis_title="Eficiencia (%)",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig_eff_stage, use_container_width=True)

        # Eficiencia global Stage 4 vs Stage 8
        st.markdown("### 🎯 Eficiencia Global: Empty (Stage 4) vs Loaded (Stage 8)")

        overall = efficiency_stats["overall_efficiency"]
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.metric("🚛 Empty (Stage 4)", f"{overall['StageSequence_4']:.2f}%")
        with col2:
            st.metric("📦 Loaded (Stage 8)", f"{overall['StageSequence_8']:.2f}%")
        with col3:
            diff = abs(overall["StageSequence_4"] - overall["StageSequence_8"])
            st.metric("📊 Diferencia", f"{diff:.2f}%")

    # TAB 3: EFFICIENCY BY CYCLE AND OPERATION
    with tab3:
        st.subheader("🔄 Eficiencia Agrupada por Ciclo y Operación")

        cycle_summary = pl.DataFrame(efficiency_stats["cycle_group_summary"])

        # filter
        col1, col2 = st.columns(2)
        with col1:
            operation_filter = st.selectbox(
                "Filtrar por Operación:", options=["Todas", "Empty", "Loaded"]
            )

        # apply filters
        if operation_filter != "Todas":
            filtered_summary = cycle_summary.filter(
                pl.col("OperationGroup") == operation_filter
            )
        else:
            filtered_summary = cycle_summary

        # show metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Registros", f"{len(filtered_summary):,}")
        with col2:
            avg_total_eff = filtered_summary["TotalEfficiency"].mean()
            if avg_total_eff is not None:
                st.metric("📈 Eficiencia Total Media", f"{avg_total_eff:.1f}%")
            else:
                st.metric("📈 Eficiencia Total Media", "N/A")
        with col3:
            avg_tonnage = filtered_summary["MeasuredTonnage"].mean()
            if avg_tonnage is not None:
                st.metric("⚖️ Tonelaje Medio", f"{avg_tonnage:.1f} ton")
            else:
                st.metric("⚖️ Tonelaje Medio", "N/A")
        with col4:
            avg_distance = filtered_summary["Distance"].mean()
            if avg_distance is not None:
                st.metric("📏 Distancia Media", f"{avg_distance:.0f} m")
            else:
                st.metric("📏 Distancia Media", "N/A")

        # Total efficiency distribution
        st.markdown("### 📊 Distribución de Eficiencia Total por Operación")
        fig_total_eff = px.histogram(
            filtered_summary.to_pandas(),
            x="TotalEfficiency",
            color="OperationGroup",
            nbins=50,
            title="Histograma de Eficiencia Total",
            labels={"TotalEfficiency": "Eficiencia Total (%)", "count": "Frecuencia"},
            color_discrete_map={"Empty": "#3498db", "Loaded": "#e74c3c"},
        )
        fig_total_eff.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_total_eff, use_container_width=True)

        # Box plot
        fig_box_eff = px.box(
            filtered_summary.to_pandas(),
            x="OperationGroup",
            y="TotalEfficiency",
            color="OperationGroup",
            title="Comparación de Eficiencia: Empty vs Loaded",
            labels={
                "TotalEfficiency": "Eficiencia Total (%)",
                "OperationGroup": "Tipo de Operación",
            },
            color_discrete_map={"Empty": "#3498db", "Loaded": "#e74c3c"},
        )
        fig_box_eff.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_box_eff, use_container_width=True)

        # Data table
        with st.expander("📋 Ver Datos Detallados"):
            st.dataframe(
                filtered_summary.to_pandas(), use_container_width=True, height=400
            )

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: FACTORES STAGE 4 & 8
    # ═══════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("📍 Análisis de Factores por Stage 4 (Carga) y Stage 8 (Descarga)")

        # Stage 4 Factors
        st.markdown("### 🔵 Stage 4 - Factores de Carga (Empty)")

        stage4_factors = efficiency_stats["efficiency_stage4_factors"]

        # Crear tabs para cada factor
        tab4_1, tab4_2, tab4_3, tab4_4 = st.tabs(
            ["🏗️ Shovel", "🚜 Modelo Pala", "📍 Zona de Carga", "📏 Rango Distancia"]
        )

        with tab4_1:
            shovel_df = pl.DataFrame(stage4_factors["Shovel"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                shovel_df.to_pandas(),
                x="Shovel",
                y="AvgEfficiency",
                title="Eficiencia por Pala",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in shovel_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab4_2:
            model_df = pl.DataFrame(stage4_factors["ShovelModel"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                model_df.to_pandas(),
                x="ShovelModel",
                y="AvgEfficiency",
                title="Eficiencia por Modelo de Pala",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in model_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab4_3:
            zone_df = pl.DataFrame(stage4_factors["LoadingZone"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                zone_df.to_pandas(),
                x="LoadingZone",
                y="AvgEfficiency",
                title="Eficiencia por Zona de Carga",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in zone_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab4_4:
            dist_df = pl.DataFrame(stage4_factors["DistanceRange"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                dist_df.to_pandas(),
                x="DistanceRange",
                y="AvgEfficiency",
                title="Eficiencia por Rango de Distancia",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in dist_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Stage 8 Factors
        st.markdown("### 🔴 Stage 8 - Factores de Descarga (Loaded)")

        stage8_factors = efficiency_stats["efficiency_stage8_factors"]

        tab8_1, tab8_2, tab8_3, tab8_4, tab8_5, tab8_6 = st.tabs(
            [
                "🪨 Material",
                "⚖️ Tonelaje Medido",
                "📊 Tonelaje Reportado",
                "🎯 Tipo Destino",
                "📍 Destino",
                "📏 Rango Distancia",
            ]
        )

        with tab8_1:
            material_df = pl.DataFrame(stage8_factors["Material"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                material_df.to_pandas(),
                x="Material",
                y="AvgEfficiency",
                title="Eficiencia por Material",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in material_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab8_2:
            mton_df = pl.DataFrame(stage8_factors["MeasuredTonnageRange"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                mton_df.to_pandas(),
                x="MeasuredTonnageRange",
                y="AvgEfficiency",
                title="Eficiencia por Rango de Tonelaje Medido",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in mton_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab8_3:
            rton_df = pl.DataFrame(stage8_factors["ReportedTonnageRange"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                rton_df.to_pandas(),
                x="ReportedTonnageRange",
                y="AvgEfficiency",
                title="Eficiencia por Rango de Tonelaje Reportado",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in rton_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab8_4:
            dtype_df = pl.DataFrame(stage8_factors["DestinationType"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                dtype_df.to_pandas(),
                x="DestinationType",
                y="AvgEfficiency",
                title="Eficiencia por Tipo de Destino",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in dtype_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab8_5:
            dest_df = (
                pl.DataFrame(stage8_factors["Destination"])
                .sort("AvgEfficiency", descending=True)
                .head(15)
            )
            fig = px.bar(
                dest_df.to_pandas(),
                x="Destination",
                y="AvgEfficiency",
                title="Eficiencia por Destino (Top 15)",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in dest_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab8_6:
            dist8_df = pl.DataFrame(stage8_factors["DistanceRange"]).sort(
                "AvgEfficiency", descending=True
            )
            fig = px.bar(
                dist8_df.to_pandas(),
                x="DistanceRange",
                y="AvgEfficiency",
                title="Eficiencia por Rango de Distancia",
                color="AvgEfficiency",
                color_continuous_scale="RdYlGn",
                text=[f"{val:.1f}%" for val in dist8_df["AvgEfficiency"].to_list()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 5: PATRONES TEMPORALES
    # ═══════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("📅 Análisis de Patrones Temporales")

        eff_by_time = efficiency_stats["efficiency_by_time"]

        # Eficiencia por día de la semana
        st.markdown("### 📆 Eficiencia por Día de la Semana")
        weekday_df = pl.DataFrame(eff_by_time["weekday_efficiency"])

        # Mapeo de días
        day_names = {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo",
        }
        weekday_df = weekday_df.with_columns(
            pl.col("Weekday")
            .map_elements(lambda x: day_names.get(x, str(x)), return_dtype=pl.String)
            .alias("DayName")
        )

        fig_weekday = px.line(
            weekday_df.to_pandas(),
            x="DayName",
            y="AvgEfficiency",
            color="OperationGroup",
            markers=True,
            title="Eficiencia por Día de la Semana",
            labels={"AvgEfficiency": "Eficiencia (%)", "DayName": "Día"},
            color_discrete_map={"Empty": "#3498db", "Loaded": "#e74c3c"},
        )
        fig_weekday.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_weekday, use_container_width=True)

        # Eficiencia por hora del día
        st.markdown("### ⏰ Eficiencia por Hora del Día")
        hour_df = pl.DataFrame(eff_by_time["hour_efficiency"])

        fig_hour = px.line(
            hour_df.to_pandas(),
            x="Hour",
            y="AvgEfficiency",
            color="OperationGroup",
            markers=True,
            title="Eficiencia por Hora del Día",
            labels={"AvgEfficiency": "Eficiencia (%)", "Hour": "Hora"},
            color_discrete_map={"Empty": "#3498db", "Loaded": "#e74c3c"},
        )
        fig_hour.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_hour, use_container_width=True)

        # Eficiencia por mes
        st.markdown("### 📅 Eficiencia por Mes")
        month_df = pl.DataFrame(eff_by_time["month_efficiency"])

        # Mapeo de meses
        month_names = {
            1: "Ene",
            2: "Feb",
            3: "Mar",
            4: "Abr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Ago",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dic",
        }
        month_df = month_df.with_columns(
            pl.col("Month")
            .map_elements(lambda x: month_names.get(x, str(x)), return_dtype=pl.String)
            .alias("MonthName")
        )

        fig_month = px.line(
            month_df.to_pandas(),
            x="MonthName",
            y="AvgEfficiency",
            color="OperationGroup",
            markers=True,
            title="Eficiencia por Mes",
            labels={"AvgEfficiency": "Eficiencia (%)", "MonthName": "Mes"},
            color_discrete_map={"Empty": "#3498db", "Loaded": "#e74c3c"},
        )
        fig_month.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_month, use_container_width=True)

        # Heatmap hora vs día
        st.markdown("### 🔥 Heatmap: Eficiencia por Hora y Día")

        # Preparar datos para heatmap
        heatmap_data = df.group_by(["Hour", "Weekday"]).agg(
            pl.col("TimeEfficiencyPercentage").mean().alias("AvgEfficiency")
        )

        # Crear matriz pivot
        pivot_data = heatmap_data.pivot(
            values="AvgEfficiency", index="Hour", on="Weekday"
        ).sort("Hour")

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=pivot_data.select(pl.all().exclude("Hour")).to_numpy(),
                x=[day_names.get(i, str(i)) for i in range(7)],
                y=pivot_data["Hour"].to_list(),
                colorscale="RdYlGn",
                colorbar=dict(title="Eficiencia (%)"),
                hoverongaps=False,
            )
        )

        fig_heatmap.update_layout(
            title="Heatmap de Eficiencia: Hora vs Día de la Semana",
            xaxis_title="Día de la Semana",
            yaxis_title="Hora del Día",
            height=600,
            template="plotly_white",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 6: ESTADÍSTICAS DETALLADAS
    # ═══════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("📈 Estadísticas Detalladas por Variable")

        # Selector de variable
        available_vars = [
            k
            for k in stats.keys()
            if isinstance(stats[k], dict)
            and stats[k].get("type") in ["categorical", "numeric"]
        ]

        selected_var = st.selectbox("Seleccionar Variable:", options=available_vars)

        if selected_var:
            var_stats = stats[selected_var]
            var_type = var_stats.get("type")

            if var_type == "categorical":
                st.markdown(f"### 📊 Variable Categórica: **{selected_var}**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Valores Únicos", var_stats.get("total_unique_values", "N/A")
                    )
                with col2:
                    st.metric("Valores Nulos", var_stats.get("null_count", "N/A"))
                with col3:
                    stage_filter = var_stats.get("stage_filter", "sin filtro")
                    st.info(f"**Filtro:** {stage_filter}")

                # Gráfico de barras
                values_df = pl.DataFrame(var_stats.get("values", [])).head(20)

                if len(values_df) > 0:
                    fig = px.bar(
                        values_df.to_pandas(),
                        x=selected_var,
                        y="count",
                        title=f"Top 20 Valores de {selected_var}",
                        color="count",
                        color_continuous_scale="Viridis",
                        text="count",
                    )
                    fig.update_layout(height=500, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                # Tabla completa
                with st.expander("📋 Ver Todos los Valores"):
                    full_values_df = pl.DataFrame(var_stats.get("values", []))
                    st.dataframe(full_values_df.to_pandas(), use_container_width=True)

            elif var_type == "numeric":
                st.markdown(f"### 📈 Variable Numérica: **{selected_var}**")

                stage_filter = var_stats.get("stage_filter", "sin filtro")
                st.info(f"**Filtro aplicado:** {stage_filter}")

                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Media", f"{var_stats.get('mean', 0):.2f}")
                with col2:
                    st.metric("Mediana", f"{var_stats.get('median', 0):.2f}")
                with col3:
                    st.metric("Desv. Est.", f"{var_stats.get('std_dev', 0):.2f}")
                with col4:
                    cv = var_stats.get("cv", 0)
                    if cv:
                        st.metric("Coef. Variación", f"{cv:.3f}")

                # Métricas adicionales
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Mínimo", f"{var_stats.get('min', 0):.2f}")
                with col2:
                    st.metric("Q1 (25%)", f"{var_stats.get('q1', 0):.2f}")
                with col3:
                    st.metric("Q3 (75%)", f"{var_stats.get('q3', 0):.2f}")
                with col4:
                    st.metric("Máximo", f"{var_stats.get('max', 0):.2f}")
                with col5:
                    st.metric("Registros", f"{var_stats.get('non_null_count', 0):,}")

                # Estadísticas avanzadas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("P5", f"{var_stats.get('p5', 0):.2f}")
                with col2:
                    st.metric("P95", f"{var_stats.get('p95', 0):.2f}")
                with col3:
                    skew = var_stats.get("skewness", 0)
                    st.metric("Asimetría", f"{skew:.3f}" if skew else "N/A")
                with col4:
                    kurt = var_stats.get("kurtosis", 0)
                    st.metric("Curtosis", f"{kurt:.3f}" if kurt else "N/A")

                # Obtener datos filtrados según el stage
                stage_filter_text = var_stats.get("stage_filter", "sin filtro")

                if "StageSequence == 8" in stage_filter_text:
                    plot_df = df.filter(pl.col("StageSequence") == 8)
                elif "StageSequence == 4" in stage_filter_text:
                    plot_df = df.filter(pl.col("StageSequence") == 4)
                elif "StageSequence == 4 OR 8" in stage_filter_text:
                    plot_df = df.filter(pl.col("StageSequence").is_in([4, 8]))
                else:
                    plot_df = df

                # Histograma
                st.markdown("#### 📊 Distribución")
                fig_hist = px.histogram(
                    plot_df.to_pandas(),
                    x=selected_var,
                    nbins=50,
                    title=f"Histograma de {selected_var}",
                    color_discrete_sequence=["#2ecc71"],
                )
                fig_hist.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Box plot
                col1, col2 = st.columns(2)

                with col1:
                    fig_box = px.box(
                        plot_df.to_pandas(),
                        y=selected_var,
                        title=f"Box Plot de {selected_var}",
                        color_discrete_sequence=["#3498db"],
                    )
                    fig_box.update_layout(height=400, template="plotly_white")
                    st.plotly_chart(fig_box, use_container_width=True)

                with col2:
                    # Violin plot
                    fig_violin = px.violin(
                        plot_df.to_pandas(),
                        y=selected_var,
                        title=f"Violin Plot de {selected_var}",
                        color_discrete_sequence=["#e74c3c"],
                        box=True,
                    )
                    fig_violin.update_layout(height=400, template="plotly_white")
                    st.plotly_chart(fig_violin, use_container_width=True)

        # Estadísticas JSON completas
        st.markdown("---")
        with st.expander("🔍 Ver Estadísticas Completas (JSON)"):
            st.json(stats)

        # Análisis de eficiencia JSON
        with st.expander("📊 Ver Análisis de Eficiencia Completo (JSON)"):
            st.json(efficiency_stats)


if __name__ == "__main__":
    show()
