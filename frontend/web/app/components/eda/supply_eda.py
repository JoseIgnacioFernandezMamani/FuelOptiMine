import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import polars as pl
from analytics.EDA.fuel_supply.fuel_supply_eda import FuelSupplyEDA
import logging
from datetime import date
from mlflow_server.config import TRUCK_IDS


@st.cache_resource
def get_analyzer(truck_id: str):
    """Cargar y cachear el analizador de datos"""
    analyzer = FuelSupplyEDA(truck_id=truck_id)
    analyzer.run()
    return analyzer


def load_data(truck_id: str):
    """Cargar datos según el modo"""
    with st.spinner("Cargando datos de Fuel Supply..."):
        analyzer = get_analyzer(truck_id)
        df_supply = analyzer.get_dataframe()
        stats = analyzer.get_statistics()
        refuel_analysis = analyzer.analyze_refuel_events()
        fleet_summary = analyzer.get_fleet_summary()
        origin_summary = analyzer.get_origin_summary()
        shift_analysis = analyzer.get_shift_analysis()
        equipment_ranking = analyzer.get_equipment_ranking(top_n=20)
        temporal_patterns = analyzer.get_temporal_patterns()

    return (
        analyzer,
        df_supply,
        stats,
        refuel_analysis,
        fleet_summary,
        origin_summary,
        shift_analysis,
        equipment_ranking,
        temporal_patterns,
    )


def render_kpi_cards(stats: dict, refuel_analysis: dict):
    """Tarjetas de KPIs principales"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_refuels = refuel_analysis.get("total_refuels", 0)
        st.metric("🔢 Total Recargas", f"{total_refuels:,}")

    with col2:
        total_volume = refuel_analysis.get("total_volume_liters", 0)
        st.metric("⛽ Volumen Total", f"{total_volume:,.0f} L")

    with col3:
        avg_volume = refuel_analysis.get("avg_volume_liters", 0)
        st.metric("📊 Promedio/Recarga", f"{avg_volume:,.0f} L")

    with col4:
        avg_fuel = stats.get("FuelLevelLiters", {}).get("mean", 0)
        st.metric("⚡ Nivel Promedio", f"{avg_fuel:,.0f} L")


def plot_refuels_by_shift(data: list):
    """Gráfico de recargas por turno"""
    if not data:
        st.warning("No hay datos de recargas por turno")
        return

    df = pd.DataFrame(data)

    fig = go.Figure()

    colors = {"D": "#ff7f0e", "N": "#1f77b4"}

    fig.add_trace(
        go.Bar(
            x=df["Shift"],
            y=df["count"],
            text=df["count"],
            textposition="auto",
            marker_color=[colors.get(shift, "#gray") for shift in df["Shift"]],
            hovertemplate="<b>Turno %{x}</b><br>Recargas: %{y}<br>Volumen: %{customdata:,.0f} L<extra></extra>",
            customdata=df["total_volume"],
        )
    )

    fig.update_layout(
        title="Distribución de Recargas por Turno",
        xaxis_title="Turno",
        yaxis_title="Cantidad de Recargas",
        showlegend=False,
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_refuels_by_origin(data: list):
    """Gráfico de recargas por origen"""
    if not data:
        st.warning("No hay datos de recargas por origen")
        return

    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        values="count",
        names="Origin",
        title="Distribución de Recargas por Origen",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Recargas: %{value}<br>%{percent}<extra></extra>",
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_volume_by_origin(data: list):
    """Gráfico de volumen por origen"""
    if not data:
        st.warning("No hay datos")
        return

    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Origin"],
            y=df["total_volume"],
            text=[f"{v:,.0f} L" for v in df["total_volume"]],
            textposition="auto",
            marker_color="#2ca02c",
            hovertemplate="<b>%{x}</b><br>Volumen: %{y:,.0f} L<br>Promedio: %{customdata:,.0f} L<extra></extra>",
            customdata=df["avg_volume"],
        )
    )

    fig.update_layout(
        title="Volumen Total Suministrado por Origen",
        xaxis_title="Origen",
        yaxis_title="Litros",
        showlegend=False,
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_trend(data: list):
    """Gráfico de tendencia mensual"""
    if not data:
        st.warning("No hay datos de tendencia mensual")
        return

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
    df = df.sort_values("Date")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["count"],
            name="Cantidad",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["total_volume"],
            name="Volumen (L)",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Mes")
    fig.update_yaxes(title_text="Cantidad de Recargas", secondary_y=False)
    fig.update_yaxes(title_text="Volumen (Litros)", secondary_y=True)

    fig.update_layout(
        title="Evolución Mensual de Recargas",
        hovermode="x unified",
        height=450,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_distribution(data: list):
    """Gráfico de distribución horaria"""
    if not data:
        st.warning("No hay datos")
        return

    df = pd.DataFrame(data)
    df = df.sort_values("Hour")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Hour"],
            y=df["count"],
            marker_color="#9467bd",
            hovertemplate="<b>Hora %{x}:00</b><br>Recargas: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Distribución Horaria de Recargas",
        xaxis_title="Hora del Día",
        yaxis_title="Cantidad de Recargas",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        showlegend=False,
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_fuel_level_timeline(df_polars: pl.DataFrame):
    """Gráfico de timeline de nivel de combustible"""
    df = df_polars.to_pandas()

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df["TimeStamp"],
            y=df["FuelLevelLiters"],
            mode="lines",
            name="Nivel de Combustible",
            line=dict(color="#17becf", width=1),
            fill="tozeroy",
            fillcolor="rgba(23, 190, 207, 0.2)",
            hovertemplate="<b>%{x}</b><br>Nivel: %{y:,.0f} L<extra></extra>",
        )
    )

    fig.update_layout(
        title="Evolución del Nivel de Combustible",
        xaxis_title="Fecha",
        yaxis_title="Litros",
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_weekday_distribution(data: list):
    """Distribución por día de la semana"""
    if not data:
        st.warning("No hay datos")
        return

    df = pd.DataFrame(data)
    weekday_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    df["DayName"] = df["Weekday"].apply(lambda x: weekday_names[x - 1])

    fig = px.bar(
        df,
        x="DayName",
        y="count",
        title="Distribución por Día de la Semana",
        labels={"count": "Cantidad de Registros", "DayName": "Día"},
        color="count",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_equipment_ranking(data: list):
    """Ranking de equipos"""
    if not data:
        st.warning("No hay datos")
        return

    df = pd.DataFrame(data)

    fig = px.bar(
        df.head(10),
        x="Equipment",
        y="total_records",
        color="avg_fuel_level",
        title="Top 10 Equipos por Cantidad de Registros",
        labels={
            "total_records": "Total Registros",
            "avg_fuel_level": "Nivel Promedio (L)",
        },
        color_continuous_scale="Viridis",
    )

    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)


def show():
    """Función principal del dashboard"""

    with st.sidebar:
        st.header("🚚 Selección de Camión")
        truck_id = st.selectbox(
            "Camión:",
            options=TRUCK_IDS,
            index=0,
        )

    st.title(f"⛽ Análisis Fuel Supply - {truck_id}")

    try:
        (
            analyzer,
            df_supply,
            stats,
            refuel_analysis,
            fleet_summary,
            origin_summary,
            shift_analysis,
            equipment_ranking,
            temporal_patterns,
        ) = load_data(truck_id)

        # KPIs principales
        render_kpi_cards(stats, refuel_analysis)
        st.markdown("---")

        # Tabs principales
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📋 Resumen General",
                "⛽ Análisis de Recargas",
                "📈 Tendencias Temporales",
                "📉 Timeline de Combustible",
                "🔍 Correlación de Eventos",
                "📊 Estadísticas Detalladas",
            ]
        )

        # ═══════════════════════════════════════════════════════════════
        # TAB 1: RESUMEN GENERAL
        # ═══════════════════════════════════════════════════════════════
        with tab1:
            st.subheader("📋 Información General")

            # Información temporal
            if "temporal" in stats:
                temporal = stats["temporal"]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📅 Días de Análisis", f"{temporal['total_days']}")
                    st.caption(f"Desde: {temporal['first_record']}")

                with col2:
                    st.metric("📊 Total Registros", f"{len(df_supply):,}")
                    st.caption(f"Hasta: {temporal['last_record']}")

                with col3:
                    st.metric("📆 Fechas Únicas", f"{temporal['unique_dates']}")

            st.markdown("---")

            # Distribución por turno y origen
            col1, col2 = st.columns(2)

            with col1:
                plot_refuels_by_shift(refuel_analysis.get("refuels_by_shift", []))

            with col2:
                plot_refuels_by_origin(refuel_analysis.get("refuels_by_origin", []))

            st.markdown("---")

            # Análisis de turnos
            st.subheader("🌓 Comparativa de Turnos")
            if shift_analysis and "shift_analysis" in shift_analysis:
                shift_df = pd.DataFrame(shift_analysis["shift_analysis"])
                st.dataframe(shift_df, use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 2: ANÁLISIS DE RECARGAS
        # ═══════════════════════════════════════════════════════════════
        with tab2:
            st.subheader("⛽ Análisis Detallado de Recargas")

            col1, col2 = st.columns(2)

            with col1:
                plot_volume_by_origin(refuel_analysis.get("refuels_by_origin", []))

            with col2:
                plot_hourly_distribution(refuel_analysis.get("refuels_by_hour", []))

            st.markdown("---")

            # Ranking de equipos
            st.subheader("🏆 Ranking de Equipos")
            plot_equipment_ranking(equipment_ranking.get("equipment_ranking", []))

            st.markdown("---")

            # Tabla de equipos
            if equipment_ranking and "equipment_ranking" in equipment_ranking:
                equip_df = pd.DataFrame(equipment_ranking["equipment_ranking"])
                if not equip_df.empty:
                    st.dataframe(equip_df, use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 3: TENDENCIAS TEMPORALES
        # ═══════════════════════════════════════════════════════════════
        with tab3:
            st.subheader("📈 Tendencias Temporales")

            plot_monthly_trend(refuel_analysis.get("refuels_by_month", []))

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                plot_weekday_distribution(
                    temporal_patterns.get("weekday_distribution", [])
                )

            with col2:
                if origin_summary and "origin_summary" in origin_summary:
                    origin_df = pd.DataFrame(origin_summary["origin_summary"])
                    if not origin_df.empty:
                        fig = px.bar(
                            origin_df,
                            x="Origin",
                            y="total_records",
                            title="Frecuencia de Uso por Origen",
                            labels={"total_records": "Total Registros"},
                            color="total_records",
                            color_continuous_scale="Blues",
                        )
                        fig.update_layout(template="plotly_white", height=400)
                        st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 4: TIMELINE DE COMBUSTIBLE
        # ═══════════════════════════════════════════════════════════════
        with tab4:
            st.subheader("📉 Timeline Completo de Nivel de Combustible")
            plot_fuel_level_timeline(df_supply)

            st.markdown("---")

            # Filtro por rango de fechas
            st.subheader("🔍 Filtrar por Rango de Fechas")

            col1, col2 = st.columns(2)
            df_pandas = df_supply.to_pandas()

            min_date = df_supply.select(pl.col("ShiftDate")).min().item()
            max_date = df_supply.select(pl.col("ShiftDate")).max().item()

            with col1:
                start_date = st.date_input(
                    "Fecha Inicio",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                )

            with col2:
                end_date = st.date_input(
                    "Fecha Fin", value=max_date, min_value=min_date, max_value=max_date
                )

            # Filtrar datos
            mask = (df_pandas["ShiftDate"] >= pd.Timestamp(start_date)) & (
                df_pandas["ShiftDate"] <= pd.Timestamp(end_date)
            )
            filtered_df = df_pandas[mask]

            if not filtered_df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scattergl(
                        x=filtered_df["TimeStamp"],
                        y=filtered_df["FuelLevelLiters"],
                        mode="lines+markers",
                        name="Nivel",
                        line=dict(color="#2ca02c", width=2),
                        marker=dict(size=4),
                    )
                )

                fig.update_layout(
                    title=f"Período: {start_date} a {end_date}",
                    xaxis_title="Fecha",
                    yaxis_title="Litros",
                    height=450,
                    template="plotly_white",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Métricas del período
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Nivel Promedio",
                        f"{filtered_df['FuelLevelLiters'].mean():,.0f} L",
                    )
                with col2:
                    st.metric(
                        "Nivel Máximo", f"{filtered_df['FuelLevelLiters'].max():,.0f} L"
                    )
                with col3:
                    st.metric(
                        "Nivel Mínimo", f"{filtered_df['FuelLevelLiters'].min():,.0f} L"
                    )
            else:
                st.warning("No hay datos en el rango seleccionado")

        # ═══════════════════════════════════════════════════════════════
        # TAB 5: CORRELACIÓN DE EVENTOS
        # ═══════════════════════════════════════════════════════════════
        with tab5:
            st.subheader("🔍 Correlación de Eventos Sensor-Supply")

            col1, col2 = st.columns(2)

            with col1:
                start_corr = st.date_input(
                    "Fecha Inicio Correlación",
                    value=date(2024, 2, 1),
                )

            with col2:
                end_corr = st.date_input(
                    "Fecha Fin Correlación",
                    value=date(2025, 2, 28),
                )

            if st.button("🔗 Ejecutar Correlación"):
                with st.spinner("Correlacionando eventos..."):
                    try:
                        correlated_df = analyzer.correlate_supply_events(
                            start_date=start_corr, end_date=end_corr
                        )

                        st.success(
                            f"✅ Correlación completada: {len(correlated_df)} eventos"
                        )

                        # Mostrar primeros registros
                        st.dataframe(
                            correlated_df.to_pandas().head(50),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Opción de descarga
                        csv = correlated_df.to_pandas().to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Correlación CSV",
                            data=csv,
                            file_name=f"correlated_events_{truck_id}_{start_corr}_{end_corr}.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"❌ Error en correlación: {str(e)}")

        # ═══════════════════════════════════════════════════════════════
        # TAB 6: ESTADÍSTICAS DETALLADAS
        # ═══════════════════════════════════════════════════════════════
        with tab6:
            st.subheader("📊 Estadísticas Descriptivas")

            # Estadísticas numéricas
            numeric_stats = {
                k: v
                for k, v in stats.items()
                if k not in ["temporal"] and isinstance(v, dict) and "mean" in v
            }

            if numeric_stats:
                rows = []
                for col, data in numeric_stats.items():
                    rows.append(
                        {
                            "Variable": col,
                            "Promedio": f"{data.get('mean', 0):,.2f}",
                            "Mediana": f"{data.get('median', 0):,.2f}",
                            "Mínimo": f"{data.get('min', 0):,.2f}",
                            "Máximo": f"{data.get('max', 0):,.2f}",
                            "Desv. Est.": f"{data.get('std', 0):,.2f}",
                        }
                    )

                stats_df = pd.DataFrame(rows)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Resumen por flota
            st.subheader("🚛 Resumen por Flota")
            if fleet_summary and "fleet_summary" in fleet_summary:
                fleet_df = pd.DataFrame(fleet_summary["fleet_summary"])
                if not fleet_df.empty:
                    st.dataframe(fleet_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Exportar datos
            st.subheader("📥 Exportar Datos")

            csv = df_supply.to_pandas().to_csv(index=False)
            st.download_button(
                label="📥 Descargar Dataset Completo",
                data=csv,
                file_name=f"fuel_supply_{truck_id}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        logging.error(f"Error en dashboard: {e}", exc_info=True)

        # Mostrar detalles del error en modo desarrollo
        with st.expander("🔍 Ver detalles del error"):
            st.code(str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    show()
