import streamlit as st
import plotly.express as px


def plot_histogram(df, column: str, title: str):
    """Componente reutilizable para histogramas"""
    fig = px.histogram(
        df.to_pandas(),
        x=column,
        nbins=50,
        title=title,
        labels={column: column.replace("_", " ").title()},
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
