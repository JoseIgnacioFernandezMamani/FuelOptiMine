import streamlit as st
import polars as pl
from analytics.EDA.time_model.time_model_data_eda import TimeModelDataEDA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from mlflow_server.config import TRUCK_IDS


# load data from database
@st.cache_resource
def get_analyzer(truck_id: str):
    """Cargar y cachear el analizador de datos"""
    analyzer = TimeModelDataEDA(truck_id=truck_id)
    analyzer.run()
    return analyzer


def load_data(truck_id: str):
    """Cargar datos según el modo"""

    with st.spinner("Cargando datos de Time Model..."):
        analyzer = get_analyzer(truck_id)
        df = analyzer.get_dataframe()
        stats = analyzer.get_statistics()
        state_patterns = analyzer.analyze_state_patterns()
        temporal_patterns = analyzer.analyze_temporal_patterns()
        uptime_analysis = analyzer.analyze_uptime_downtime()
        cat_event_rel = analyzer.analyze_category_event_relationship()
    return (
        analyzer,
        df,
        stats,
        state_patterns,
        temporal_patterns,
        uptime_analysis,
        cat_event_rel,
    )


def show():

    with st.sidebar:
        st.header("🚚 Elige un camion para analizar")
        truck_id = st.selectbox(
            "Selecciona el ID del Camión:",
            options=TRUCK_IDS,
            index=0,
        )

    st.title(f"📊 Análisis Time Model - Estados y Eventos del camion {truck_id}")

    (
        analyzer,
        df,
        stats,
        state_patterns,
        temporal_patterns,
        uptime_analysis,
        cat_event_rel,
    ) = load_data(truck_id)

    # Tabs principales
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📋 Resumen General",
            "🔄 Estados y Transiciones",
            "⏱️ Patrones Temporales",
            "⚡ Uptime/Downtime",
            "📊 Categorías y Eventos",
            "📈 Estadísticas Detalladas",
        ]
    )

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: RESUMEN GENERAL
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.subheader(f"📋 Resumen General - {truck_id}")

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_events = stats.get("TimeModelId", {}).get("total_events", 0)
            st.metric("🔢 Total Eventos", f"{total_events:,}")

        with col2:
            total_records = stats.get("TimeModelId", {}).get("total_records", 0)
            st.metric("📝 Total Registros", f"{total_records:,}")

        with col3:
            unique_statuses = stats.get("Status", {}).get("unique_values", 0)
            st.metric("📊 Estados Únicos", f"{unique_statuses}")

        with col4:
            avg_duration = stats.get("StateDurationSeconds", {}).get("mean_minutes", 0)
            if avg_duration:
                st.metric("⏱️ Duración Media", f"{avg_duration:.1f} min")

        # Rango temporal
        st.markdown("### 📅 Información Temporal")
        col1, col2, col3 = st.columns(3)

        if "TimeStamp_tm" in stats:
            with col1:
                st.info(
                    f"**Inicio:** {stats['TimeStamp_tm'].get('first_record', 'N/A')}"
                )
            with col2:
                st.info(f"**Fin:** {stats['TimeStamp_tm'].get('last_record', 'N/A')}")
            with col3:
                duration_days = stats["TimeStamp_tm"].get("total_duration_days", 0)
                st.info(f"**Duración:** {duration_days:.1f} días")

        # Distribución de estados principales
        st.markdown("### 🎯 Top 10 Estados Más Frecuentes")

        if "Status" in stats and "top_10_values" in stats["Status"]:
            status_df = pl.DataFrame(stats["Status"]["top_10_values"])

            fig_status = px.bar(
                status_df.to_pandas(),
                x="Status",
                y="count",
                title="Frecuencia de Estados",
                color="count",
                color_continuous_scale="Viridis",
                text="count",
            )
            fig_status.update_layout(height=400, template="plotly_white")
            fig_status.update_traces(textposition="outside")
            st.plotly_chart(fig_status, use_container_width=True)

        # Distribución por turno
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌓 Distribución por Turno")
            if "Shift_distribution" in stats:
                shift_df = pl.DataFrame(stats["Shift_distribution"])
                fig_shift = px.pie(
                    shift_df.to_pandas(),
                    values="count",
                    names="Shift",
                    title="Día (D) vs Noche (N)",
                    color_discrete_sequence=["#FDB813", "#1f77b4"],
                )
                st.plotly_chart(fig_shift, use_container_width=True)

        with col2:
            st.markdown("### 📊 Top 5 Categorías")
            if "Category" in stats and "top_10_values" in stats["Category"]:
                cat_df = pl.DataFrame(stats["Category"]["top_10_values"]).head(5)
                fig_cat = px.bar(
                    cat_df.to_pandas(),
                    x="Category",
                    y="count",
                    title="Categorías Más Frecuentes",
                    color="count",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_cat, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: ESTADOS Y TRANSICIONES
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🔄 Análisis de Estados y Transiciones")

        # Métricas de transiciones
        col1, col2 = st.columns(2)

        with col1:
            total_transitions = state_patterns["state_transitions"][
                "total_unique_transitions"
            ]
            st.metric("🔀 Transiciones Únicas", f"{total_transitions:,}")

        with col2:
            total_events = state_patterns["event_frequency"]["total_unique_events"]
            st.metric("📋 Eventos Únicos", f"{total_events:,}")

        # Top transiciones
        st.markdown("### 🔝 Top 20 Transiciones de Estados")

        transitions_df = pl.DataFrame(
            state_patterns["state_transitions"]["top_20_transitions"]
        )

        # Crear labels para el gráfico
        transitions_df = transitions_df.with_columns(
            (pl.col("Status") + " → " + pl.col("NextStatus")).alias("Transition")
        )

        fig_trans = px.bar(
            transitions_df.to_pandas(),
            x="Transition",
            y="count",
            title="Transiciones Más Frecuentes",
            color="percentage",
            color_continuous_scale="RdYlGn",
            hover_data=["percentage"],
            text=[f"{p:.1f}%" for p in transitions_df["percentage"].to_list()],
        )
        fig_trans.update_layout(
            height=500, template="plotly_white", xaxis_tickangle=-45
        )
        fig_trans.update_traces(textposition="outside")
        st.plotly_chart(fig_trans, use_container_width=True)

        # Diagrama de Sankey para transiciones principales
        st.markdown("### 🌊 Diagrama de Flujo de Estados (Top 15)")

        top_transitions = transitions_df.head(15)

        # Preparar datos para Sankey
        all_states = list(
            set(
                top_transitions["Status"].to_list()
                + top_transitions["NextStatus"].to_list()
            )
        )
        state_to_idx = {state: idx for idx, state in enumerate(all_states)}

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_states,
                        color="lightblue",
                    ),
                    link=dict(
                        source=[
                            state_to_idx[s] for s in top_transitions["Status"].to_list()
                        ],
                        target=[
                            state_to_idx[s]
                            for s in top_transitions["NextStatus"].to_list()
                        ],
                        value=top_transitions["count"].to_list(),
                        label=[
                            f"{c} eventos" for c in top_transitions["count"].to_list()
                        ],
                    ),
                )
            ]
        )

        fig_sankey.update_layout(
            title="Flujo de Transiciones entre Estados", height=600, font_size=10
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

        # Duración por estado
        st.markdown("### ⏱️ Duración Acumulada por Estado")

        duration_status_df = pl.DataFrame(state_patterns["duration_by_status"]).head(15)

        col1, col2 = st.columns(2)

        with col1:
            fig_dur_hours = px.bar(
                duration_status_df.to_pandas(),
                x="Status",
                y="total_hours",
                title="Tiempo Total por Estado (Top 15)",
                color="total_hours",
                color_continuous_scale="Reds",
                text=[f"{h:.1f}h" for h in duration_status_df["total_hours"].to_list()],
            )
            fig_dur_hours.update_layout(
                height=400, template="plotly_white", xaxis_tickangle=-45
            )
            st.plotly_chart(fig_dur_hours, use_container_width=True)

        with col2:
            fig_dur_avg = px.bar(
                duration_status_df.to_pandas(),
                x="Status",
                y="avg_minutes",
                title="Duración Promedio por Estado (Top 15)",
                color="avg_minutes",
                color_continuous_scale="Blues",
                text=[f"{m:.1f}m" for m in duration_status_df["avg_minutes"].to_list()],
            )
            fig_dur_avg.update_layout(
                height=400, template="plotly_white", xaxis_tickangle=-45
            )
            st.plotly_chart(fig_dur_avg, use_container_width=True)

        # Estados más largos
        st.markdown("### 📏 Top 20 Estados Más Largos Registrados")

        longest_df = pl.DataFrame(state_patterns["longest_states"])

        # Formatear timestamp
        longest_display = longest_df.select(
            [
                pl.col("TimeStamp_tm")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
                .alias("Timestamp"),
                "Status",
                "Category",
                "Event",
                "DurationHours",
            ]
        )

        st.dataframe(longest_display.to_pandas(), use_container_width=True, height=400)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: PATRONES TEMPORALES
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("⏱️ Análisis de Patrones Temporales")

        # Distribución por hora del día
        st.markdown("### 🕐 Distribución de Estados por Hora del Día")

        hourly_df = pl.DataFrame(temporal_patterns["hourly_status_distribution"])

        # Top 5 estados para visualización
        top_5_statuses = (
            hourly_df.group_by("Status")
            .agg(pl.col("count").sum().alias("total"))
            .sort("total", descending=True)
            .head(5)["Status"]
            .to_list()
        )

        hourly_filtered = hourly_df.filter(pl.col("Status").is_in(top_5_statuses))

        fig_hourly = px.line(
            hourly_filtered.to_pandas(),
            x="Hour",
            y="count",
            color="Status",
            title="Top 5 Estados por Hora del Día",
            markers=True,
            labels={"count": "Frecuencia", "Hour": "Hora"},
        )
        fig_hourly.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_hourly, use_container_width=True)

        # Heatmap hora vs status
        st.markdown("### 🔥 Heatmap: Estados vs Hora del Día")

        # Preparar datos para heatmap (Top 10 estados)
        top_10_statuses = (
            hourly_df.group_by("Status")
            .agg(pl.col("count").sum().alias("total"))
            .sort("total", descending=True)
            .head(10)["Status"]
            .to_list()
        )

        heatmap_data = hourly_df.filter(pl.col("Status").is_in(top_10_statuses))

        pivot_data = heatmap_data.pivot(values="count", index="Status", on="Hour")

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=pivot_data.select(pl.all().exclude("Status")).to_numpy(),
                x=[str(i) for i in range(24)],
                y=pivot_data["Status"].to_list(),
                colorscale="YlOrRd",
                colorbar=dict(title="Frecuencia"),
                hoverongaps=False,
            )
        )

        fig_heatmap.update_layout(
            title="Heatmap de Estados por Hora (Top 10 Estados)",
            xaxis_title="Hora del Día",
            yaxis_title="Estado",
            height=500,
            template="plotly_white",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Distribución por día de la semana
        st.markdown("### 📅 Distribución por Día de la Semana")

        weekday_df = pl.DataFrame(temporal_patterns["weekday_status_distribution"])

        # Mapeo de días
        day_names = {
            0: "Lun",
            1: "Mar",
            2: "Mié",
            3: "Jue",
            4: "Vie",
            5: "Sáb",
            6: "Dom",
        }

        weekday_filtered = weekday_df.filter(pl.col("Status").is_in(top_5_statuses))
        weekday_filtered = weekday_filtered.with_columns(
            pl.col("Weekday")
            .map_elements(lambda x: day_names.get(x, str(x)), return_dtype=pl.String)
            .alias("DayName")
        )

        fig_weekday = px.bar(
            weekday_filtered.to_pandas(),
            x="DayName",
            y="count",
            color="Status",
            title="Top 5 Estados por Día de la Semana",
            barmode="stack",
            labels={"count": "Frecuencia", "DayName": "Día"},
        )
        fig_weekday.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_weekday, use_container_width=True)

        # Resumen diario
        st.markdown("### 📊 Resumen Diario de Actividad")

        daily_df = pl.DataFrame(temporal_patterns["daily_summary"])

        fig_daily = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Total de Eventos por Día", "Horas Totales por Día"),
            vertical_spacing=0.15,
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["Date"].to_list(),
                y=daily_df["total_events"].to_list(),
                mode="lines+markers",
                name="Eventos",
                line=dict(color="#3498db", width=2),
            ),
            row=1,
            col=1,
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["Date"].to_list(),
                y=daily_df["total_hours"].to_list(),
                mode="lines+markers",
                name="Horas",
                line=dict(color="#e74c3c", width=2),
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )

        fig_daily.update_layout(height=600, template="plotly_white", showlegend=False)
        fig_daily.update_xaxes(title_text="Fecha", row=2, col=1)
        fig_daily.update_yaxes(title_text="Eventos", row=1, col=1)
        fig_daily.update_yaxes(title_text="Horas", row=2, col=1)

        st.plotly_chart(fig_daily, use_container_width=True)

        # Distribución por turno
        st.markdown("### 🌓 Análisis por Turno")

        shift_df = pl.DataFrame(temporal_patterns["shift_status_distribution"])

        col1, col2 = st.columns(2)

        with col1:
            shift_top = shift_df.sort("count", descending=True).head(10)
            fig_shift = px.bar(
                shift_top.to_pandas(),
                x="Status",
                y="count",
                color="Shift",
                title="Top 10 Estados por Turno",
                barmode="group",
                color_discrete_map={"D": "#FDB813", "N": "#1f77b4"},
            )
            fig_shift.update_layout(
                height=400, template="plotly_white", xaxis_tickangle=-45
            )
            st.plotly_chart(fig_shift, use_container_width=True)

        with col2:
            # Porcentaje por turno
            shift_summary = shift_df.group_by("Shift").agg(
                pl.col("count").sum().alias("total")
            )
            fig_shift_pie = px.pie(
                shift_summary.to_pandas(),
                values="total",
                names="Shift",
                title="Distribución Total por Turno",
                color_discrete_map={"D": "#FDB813", "N": "#1f77b4"},
            )
            st.plotly_chart(fig_shift_pie, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: UPTIME/DOWNTIME
    # ═══════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("⚡ Análisis de Uptime y Downtime")

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_hours = uptime_analysis["total_hours"]
            st.metric("⏱️ Total Horas", f"{total_hours:.1f} h")

        with col2:
            uptime_hours = uptime_analysis["uptime_hours"]
            st.metric("✅ Uptime", f"{uptime_hours:.1f} h")

        with col3:
            downtime_hours = uptime_analysis["downtime_hours"]
            st.metric("❌ Downtime", f"{downtime_hours:.1f} h")

        with col4:
            uptime_pct = uptime_analysis["uptime_percentage"]
            st.metric("📈 % Uptime", f"{uptime_pct:.1f}%")

        # Gráfico de disponibilidad
        st.markdown("### 📊 Disponibilidad del Equipo")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            availability_data = pd.DataFrame(
                {
                    "Type": ["Uptime", "Downtime"],
                    "Hours": [uptime_hours, downtime_hours],
                    "Percentage": [
                        uptime_analysis["uptime_percentage"],
                        uptime_analysis["downtime_percentage"],
                    ],
                }
            )

            fig_avail = px.pie(
                availability_data,
                values="Hours",
                names="Type",
                title="Distribución Uptime vs Downtime",
                color="Type",
                color_discrete_map={"Uptime": "#27ae60", "Downtime": "#e74c3c"},
            )
            fig_avail.update_traces(textinfo="label+percent", textfont_size=14)
            st.plotly_chart(fig_avail, use_container_width=True)

        with col2:
            # Bar chart
            fig_avail_bar = px.bar(
                availability_data,
                x="Type",
                y="Hours",
                title="Horas de Uptime vs Downtime",
                color="Type",
                color_discrete_map={"Uptime": "#27ae60", "Downtime": "#e74c3c"},
                text=[
                    f"{h:.1f}h ({p:.1f}%)"
                    for h, p in zip(
                        availability_data["Hours"], availability_data["Percentage"]
                    )
                ],
            )
            fig_avail_bar.update_layout(
                height=400, template="plotly_white", showlegend=False
            )
            fig_avail_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_avail_bar, use_container_width=True)

        # Desglose de downtime
        st.markdown("### 🔍 Desglose de Downtime por Estado")

        downtime_df = pl.DataFrame(uptime_analysis["downtime_by_status"]).head(15)

        fig_downtime = px.bar(
            downtime_df.to_pandas(),
            x="Status",
            y="total_hours",
            title="Top 15 Causas de Downtime",
            color="total_hours",
            color_continuous_scale="Reds",
            text=[f"{h:.1f}h" for h in downtime_df["total_hours"].to_list()],
            hover_data=["occurrences"],
        )
        fig_downtime.update_layout(
            height=500, template="plotly_white", xaxis_tickangle=-45
        )
        fig_downtime.update_traces(textposition="outside")
        st.plotly_chart(fig_downtime, use_container_width=True)

        # Tabla detallada de downtime
        with st.expander("📋 Ver Desglose Completo de Downtime"):
            downtime_full = pl.DataFrame(uptime_analysis["downtime_by_status"])
            downtime_display = downtime_full.with_columns(
                [
                    (
                        pl.col("total_hours") / downtime_full["total_hours"].sum() * 100
                    ).alias("percentage")
                ]
            )
            st.dataframe(
                downtime_display.to_pandas(), use_container_width=True, height=400
            )

        # Configuración de estados operativos
        st.markdown("---")
        st.markdown("### ⚙️ Configuración de Estados Operativos")

        with st.expander("ℹ️ Estados considerados como Uptime"):
            st.info(
                """
            Estados operativos por defecto:
            - Operating
            - Running
            - Active
            - Working
            - Loaded
            - Empty
            - Hauling
            - Loading
            
            *Estos pueden ser modificados en el código según la definición de su operación.*
            """
            )

    # ═══════════════════════════════════════════════════════════════
    # TAB 5: CATEGORÍAS Y EVENTOS
    # ═══════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("📊 Análisis de Categorías y Eventos")

        # Resumen de categorías
        st.markdown("### 📋 Distribución de Categorías")

        if "Category" in stats and "all_values" in stats["Category"]:
            category_all_df = pl.DataFrame(stats["Category"]["all_values"])

            fig_cat_all = px.bar(
                category_all_df.to_pandas(),
                x="Category",
                y="count",
                title="Todas las Categorías",
                color="count",
                color_continuous_scale="Viridis",
                text="count",
            )
            fig_cat_all.update_layout(
                height=400, template="plotly_white", xaxis_tickangle=-45
            )
            fig_cat_all.update_traces(textposition="outside")
            st.plotly_chart(fig_cat_all, use_container_width=True)

        # Eventos por categoría
        st.markdown("### 🔗 Relación Categoría-Evento")

        events_per_cat_df = pl.DataFrame(cat_event_rel["events_per_category"]).head(30)

        # Crear label combinado
        events_per_cat_df = events_per_cat_df.with_columns(
            (pl.col("Category") + " - " + pl.col("Event")).alias("CategoryEvent")
        )

        fig_cat_event = px.bar(
            events_per_cat_df.to_pandas(),
            x="CategoryEvent",
            y="occurrences",
            title="Top 30 Eventos por Categoría",
            color="Category",
            text="occurrences",
            hover_data=["avg_duration_seconds"],
        )
        fig_cat_event.update_layout(
            height=500, template="plotly_white", xaxis_tickangle=-45
        )
        fig_cat_event.update_traces(textposition="outside")
        st.plotly_chart(fig_cat_event, use_container_width=True)

        # Sunburst chart
        st.markdown("### 🌅 Jerarquía Categoría-Evento")

        events_sunburst_df = pl.DataFrame(cat_event_rel["events_per_category"]).head(50)

        fig_sunburst = px.sunburst(
            events_sunburst_df.to_pandas(),
            path=["Category", "Event"],
            values="occurrences",
            title="Distribución Jerárquica (Top 50)",
            color="occurrences",
            color_continuous_scale="RdYlGn",
        )
        fig_sunburst.update_layout(height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)

        # Relación Status-Category
        st.markdown("### 🔀 Relación Status-Categoría")

        status_cat_df = pl.DataFrame(
            cat_event_rel["status_category_relationship"]
        ).head(30)

        fig_status_cat = px.bar(
            status_cat_df.to_pandas(),
            x="Status",
            y="occurrences",
            color="Category",
            title="Top 30 Combinaciones Status-Categoría",
            barmode="stack",
        )
        fig_status_cat.update_layout(
            height=500, template="plotly_white", xaxis_tickangle=-45
        )
        st.plotly_chart(fig_status_cat, use_container_width=True)

        # Top eventos más frecuentes
        st.markdown("### 🏆 Top 20 Eventos Más Frecuentes")

        if "Event" in stats and "top_10_values" in stats["Event"]:
            event_df = pl.DataFrame(stats["Event"]["all_values"]).head(20)

            fig_events = px.bar(
                event_df.to_pandas(),
                y="Event",
                x="count",
                orientation="h",
                title="Eventos Más Comunes",
                color="count",
                color_continuous_scale="Blues",
                text="count",
            )
            fig_events.update_layout(height=600, template="plotly_white")
            fig_events.update_traces(textposition="outside")
            st.plotly_chart(fig_events, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 6: ESTADÍSTICAS DETALLADAS
    # ═══════════════════════════════════════════════════════════

    with tab6:
        st.subheader("📈 Estadísticas Detalladas por Variable")

        # Selector de variable
        variable = st.selectbox(
            "Seleccionar Variable:", options=["Status", "Category", "Event"]
        )

        if variable in stats and "all_values" in stats[variable]:
            var_df = pl.DataFrame(stats[variable]["all_values"])

            # Mostrar distribución
            fig = px.bar(
                var_df.to_pandas(),
                x=variable,
                y="count",
                title=f"Distribución de {variable}",
                color="count",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla de datos
            st.dataframe(var_df.to_pandas(), use_container_width=True)

        # Duración por variable
        if "StateDurationSeconds" in df.columns:
            st.markdown("### ⏱️ Duración por Variable")

            duration_stats = (
                df.filter(pl.col("StateDurationSeconds").is_not_null())
                .group_by(variable)
                .agg(
                    [
                        pl.len().alias("total_events"),
                        pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                        pl.col("StateDurationSeconds").mean().alias("avg_seconds"),
                        pl.col("StateDurationSeconds").median().alias("median_seconds"),
                    ]
                )
                .with_columns(
                    [
                        (pl.col("total_seconds") / 3600).alias("total_hours"),
                        (pl.col("avg_seconds") / 60).alias("avg_minutes"),
                    ]
                )
                .sort("total_hours", descending=True)
            )

            st.dataframe(duration_stats.to_pandas(), use_container_width=True)

            # Box plot
            fig_box = px.box(
                df.to_pandas(),
                x=variable,
                y="StateDurationSeconds",
                title=f"Distribución de Duración por {variable}",
                color=variable,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # JSON completo
        with st.expander("🔍 Ver Estadísticas JSON"):
            st.json(stats)


if __name__ == "__main__":
    show()
