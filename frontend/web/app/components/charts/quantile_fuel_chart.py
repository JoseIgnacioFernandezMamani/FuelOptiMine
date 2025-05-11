# app.py (parte adicional)
import streamlit as st
import plotly.express as px
from datetime import datetime
from fuel_analysis import QuantileFuelAnalysis


def show_quantile_analysis():
    st.header("Análisis de Cuantiles para Recargas")

    analyzer = QuantileFuelAnalysis()
    stats = analyzer.get_refill_stats()
    df_viz = analyzer.get_visualization_data()

    # Mostrar métricas
    cols = st.columns(3)
    cols[0].metric("Umbral de Detección", f"{stats['threshold']:.2f} L")
    cols[1].metric("Recargas Detectadas", stats["total_refills"])
    cols[2].metric("Mínima Recarga", f"{stats['min_refill']:.2f} L")

    # Gráfico de dispersión
    if df_viz is not None and not df_viz.is_empty():
        df_pd = df_viz.to_pandas()

        fig = px.scatter(
            df_pd,
            x=df_pd.index,
            y="DeltaFuel",
            color="EsRecarga",
            color_discrete_map={True: "#FF5722", False: "#2196F3"},
            labels={
                "x": "Índice Ordenado",
                "DeltaFuel": "Incremento de Combustible (L)",
            },
            title="Distribución de Incrementos de Combustible",
        )

        # Línea de umbral
        fig.add_hline(
            y=stats["threshold"],
            line_dash="dot",
            line_color="red",
            annotation_text=f"Umbral: {stats['threshold']:.2f} L",
            annotation_position="bottom right",
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="lightgray"),
            plot_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontraron incrementos positivos de combustible")


# Integrar en la app principal
def main():
    st.title("📊 Análisis Avanzado de Combustible")

    # Menú lateral
    analysis_type = st.sidebar.selectbox(
        "Seleccione el tipo de análisis:",
        ["Análisis Temporal", "Análisis de Cuantiles"],
    )

    if analysis_type == "Análisis Temporal":
        show_temporal_analysis()  # Tu función existente
    else:
        show_quantile_analysis()


if __name__ == "__main__":
    main()
