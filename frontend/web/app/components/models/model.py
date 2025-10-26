"""
Streamlit app for loading XGBoost models from MLflow and generating predictions.

This module:
- Loads trained models from MLflow Model Registry
- Fetches data from ClickHouse for selected truck
- Generates predictions using specialized Stage 4 and Stage 8 models
- Displays interactive visualizations of predictions vs actual values

"""

import streamlit as st
import polars as pl
import mlflow
import mlflow.xgboost
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import folium
from folium import plugins
from streamlit_folium import st_folium
from etl_core.load.utils import create_client, CH_CONFIG
from model.predictive.report_generator import add_report_generation_to_ui
from mlflow_server.config import (
    TRUCK_IDS,
    MLFLOW_TRACKING_URI,
)
from model.predictive.report_generator import add_report_generation_to_ui

# Feature definitions for each stage
FEATURES_STAGE4 = {
    "numeric": [
        "SpeedAvg",
        "Distance",
        "CycleDurationSeconds",
        "TimeEfficiencyPercentage",
    ],
    "categorical": ["Shovel", "Destination", "DestinationType"],
}

FEATURES_STAGE8 = {
    "numeric": [
        "SpeedAvg",
        "TotalMeasuredTonnage",
        "Distance",
        "CycleDurationSeconds",
        "TimeEfficiencyPercentage",
    ],
    "categorical": ["Destination", "DestinationType", "Material"],
}


@st.cache_resource
def load_models_from_mlflow(truck_id: str) -> Tuple[Optional[object], Optional[object]]:
    """
    Load Stage 4 and Stage 8 models from MLflow Model Registry.

    Args:
        truck_id: Truck identifier (e.g., "T-210")

    Returns:
        Tuple of (model_stage4, model_stage8) or (None, None) if loading fails
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Load Stage 4 model (empty truck)
        model_stage4 = mlflow.xgboost.load_model(
            f"models:/{truck_id}_stage4_fuel/Production"
        )

        # Load Stage 8 model (loaded truck)
        model_stage8 = mlflow.xgboost.load_model(
            f"models:/{truck_id}_stage8_fuel/Production"
        )

        return model_stage4, model_stage8

    except Exception as e:
        st.error(f"Error cargando modelos desde MLflow: {str(e)}")
        st.info(
            "Asegúrese de que los modelos estén registrados y promovidos a la etapa de 'Producción' en MLflow"
        )
        return None, None


@st.cache_data
def load_data_from_clickhouse(truck_id: str) -> Optional[pl.DataFrame]:
    """
    Load operational data from ClickHouse for the selected truck.

    Args:
        truck_id: Truck identifier

    Returns:
        Polars DataFrame with truck operational data
    """

    try:
        client = create_client(CH_CONFIG)

        query = f"""
        SELECT 
            Equipment,
            ShiftDate,
            TimeStamp,
            SpeedAvg,
            Distance,
            TimeEfficiencyPercentage,
            MeasuredTonnage,
            Shovel,
            Destination,
            DestinationType,
            Material,
            StageSequence,
            TimeStampIni,
            TimeStampFin,
            CycleId,
            Latitude,
            Longitude,
            Elevation,
            Latitude_cycle,
            Longitude_cycle,
            Elevation_cycle
        FROM fuel_optimine.xgboost_fuel
        WHERE Equipment = '{truck_id}'
          AND StageSequence IN (4, 8)
        ORDER BY SortTimestamp
        """

        data = client.query_df(query)
        df = pl.from_pandas(data)

        # Apply fallback logic: use cycle coordinates if available, else use sensor coordinates
        df = df.with_columns(
            [
                pl.when(
                    (pl.col("Latitude_cycle").is_not_null())
                    & (pl.col("Latitude_cycle") != 0)
                )
                .then(pl.col("Latitude_cycle"))
                .otherwise(pl.col("Latitude"))
                .alias("Latitude"),
                pl.when(
                    (pl.col("Longitude_cycle").is_not_null())
                    & (pl.col("Longitude_cycle") != 0)
                )
                .then(pl.col("Longitude_cycle"))
                .otherwise(pl.col("Longitude"))
                .alias("Longitude"),
                pl.when(
                    (pl.col("Elevation_cycle").is_not_null())
                    & (pl.col("Elevation_cycle") != 0)
                )
                .then(pl.col("Elevation_cycle"))
                .otherwise(pl.col("Elevation"))
                .alias("Elevation"),
            ]
        ).drop(["Latitude_cycle", "Longitude_cycle", "Elevation_cycle"])

        # Calculate CycleDurationSeconds and TotalMeasuredTonnage
        df = df.with_columns(
            [
                (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                .dt.total_seconds()
                .alias("CycleDurationSeconds"),
                pl.col("MeasuredTonnage").alias("TotalMeasuredTonnage"),
            ]
        )

        return df

    except Exception as e:
        st.error(f"Error cargando datos de ClickHouse: {str(e)}")
        return None


def generate_predictions(
    df: pl.DataFrame, model_stage4: object, model_stage8: object
) -> pl.DataFrame:
    """
    Generate predictions using the appropriate model for each stage.

    Args:
        df: Input DataFrame with operational data
        model_stage4: XGBoost model for Stage 4 (empty truck)
        model_stage8: XGBoost model for Stage 8 (loaded truck)

    Returns:
        DataFrame with predictions added
    """
    df_pandas = df.to_pandas()
    predictions = np.zeros(len(df_pandas))

    # Predict Stage 4 (empty truck)
    mask_4 = df_pandas["StageSequence"] == 4
    if mask_4.any():
        features = FEATURES_STAGE4["numeric"] + FEATURES_STAGE4["categorical"]
        X_stage4 = df_pandas.loc[mask_4, features].copy()

        # Convert categorical columns and validate they have data
        for cat_col in FEATURES_STAGE4["categorical"]:
            if cat_col in X_stage4.columns:
                # Remove rows with missing values in this categorical column
                valid_mask = X_stage4[cat_col].notna() & (X_stage4[cat_col] != '')
                
                # Convert to category
                X_stage4[cat_col] = X_stage4[cat_col].astype("category")
                
                # Check if category has valid values
                if len(X_stage4[cat_col].cat.categories) == 0:
                    # Add a default category
                    X_stage4[cat_col] = pd.Categorical(['Unknown'] * len(X_stage4))

        try:
            predictions[mask_4] = model_stage4.predict(X_stage4)
        except Exception as e:
            st.error(f"Error en la predicción de la Etapa 4: {str(e)}")
            st.info("Establecer las predicciones de la Etapa 4")
            predictions[mask_4] = 0

    # Predict Stage 8 (loaded truck)
    mask_8 = df_pandas["StageSequence"] == 8
    if mask_8.any():
        features = FEATURES_STAGE8["numeric"] + FEATURES_STAGE8["categorical"]
        X_stage8 = df_pandas.loc[mask_8, features].copy()

        # Convert categorical columns and validate they have data
        for cat_col in FEATURES_STAGE8["categorical"]:
            if cat_col in X_stage8.columns:
                # Remove rows with missing values in this categorical column
                valid_mask = X_stage8[cat_col].notna() & (X_stage8[cat_col] != '')
                
                if not valid_mask.all():
                    st.warning(f"Etapa 8: Columna '{cat_col}' tiene {(~valid_mask).sum()} valores faltantes. Rellenando con 'Unknown'.")
                    X_stage8.loc[~valid_mask, cat_col] = 'Unknown'
                
                # Convert to category
                X_stage8[cat_col] = X_stage8[cat_col].astype("category")
                
                # Check if category has valid values
                if len(X_stage8[cat_col].cat.categories) == 0:
                    st.error(f"Etapa 8: Columna '{cat_col}' no tiene categorías válidas después de la conversión.")
                    # Add a default category
                    X_stage8[cat_col] = pd.Categorical(['Unknown'] * len(X_stage8))

        try:
            predictions[mask_8] = model_stage8.predict(X_stage8)
        except Exception as e:
            st.error(f"Error en la predicción de la Etapa 8: {str(e)}")
            st.info("Poniendo las predicciones de la Etapa 8")
            predictions[mask_8] = 0

    # Add predictions to DataFrame (with geographic data preserved)
    result = df.with_columns([pl.Series("PredictedFuel", predictions)])

    return result

def plot_time_series_predictions_only(df: pl.DataFrame):
    """Plot time series of predicted fuel consumption."""
    df_sorted = df.sort("TimeStamp").to_pandas()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_sorted["TimeStamp"],
            y=df_sorted["PredictedFuel"],
            mode="lines+markers",
            name="Predicted Fuel",
            line=dict(color="green", width=2),
            marker=dict(size=4),
        )
    )

    # Color by stage
    colors = {4: "lightblue", 8: "orange"}
    for stage in [4, 8]:
        stage_data = df_sorted[df_sorted["StageSequence"] == stage]
        if not stage_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=stage_data["TimeStamp"],
                    y=stage_data["PredictedFuel"],
                    mode="markers",
                    marker=dict(color=colors[stage], size=8),
                    name=f"Stage {stage}",
                    showlegend=True,
                )
            )

    fig.update_layout(
        title="Predicted Fuel Consumption Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Predicted Fuel (L)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_fuel_distribution_by_stage(df: pl.DataFrame):
    """Box plot of fuel distribution by stage."""
    df_pd = df.to_pandas()
    fig = px.box(
        df_pd,
        x="StageSequence",
        y="PredictedFuel",
        color="StageSequence",
        labels={"StageSequence": "Stage", "PredictedFuel": "Predicted Fuel (L)"},
        title="Distribution of Predicted Fuel Consumption by Stage",
        category_orders={"StageSequence": [4, 8]},
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def show_prediction_summary(df: pl.DataFrame):
    """Display summary statistics table."""
    summary = (
        df.group_by("StageSequence")
        .agg(
            pl.col("PredictedFuel").mean().alias("Mean_Predicted_Fuel"),
            pl.col("PredictedFuel").std().alias("Std_Predicted_Fuel"),
            pl.col("PredictedFuel").min().alias("Min"),
            pl.col("PredictedFuel").max().alias("Max"),
            pl.len().alias("Cycles"),
        )
        .sort("StageSequence")
        .to_pandas()
    )
    summary["Stage"] = summary["StageSequence"].map(
        {4: "Empty Truck", 8: "Loaded Truck"}
    )
    summary = summary[
        ["Stage", "Cycles", "Mean_Predicted_Fuel", "Std_Predicted_Fuel", "Min", "Max"]
    ]
    st.subheader("Resumen de predicciones por etapa")
    st.dataframe(summary.round(3), use_container_width=True)


def plot_prediction_vs_numeric(df: pl.DataFrame, stage: int):
    """Scatter plots of predictions vs numeric features."""
    df_stage = df.filter(pl.col("StageSequence") == stage).to_pandas()
    if df_stage.empty:
        st.warning(f"No data for Stage {stage}")
        return

    numeric_vars = (
        FEATURES_STAGE4["numeric"] if stage == 4 else FEATURES_STAGE8["numeric"]
    )
    numeric_vars = [col for col in numeric_vars if col in df_stage.columns]
    if not numeric_vars:
        return

    vars_to_plot = numeric_vars[:4]
    cols = st.columns(len(vars_to_plot))
    for i, col in enumerate(vars_to_plot):
        with cols[i]:
            fig = px.scatter(
                df_stage,
                x=col,
                y="PredictedFuel",
                trendline="ols",
                title=f"{col} vs Predicted Fuel",
                labels={"PredictedFuel": "Predicted Fuel (L)"},
            )
            fig.update_layout(height=300, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


def plot_fuel_by_category(df: pl.DataFrame, stage: int):
    """Bar charts of fuel by categorical features."""
    df_stage = df.filter(pl.col("StageSequence") == stage).to_pandas()
    if df_stage.empty:
        return

    cat_vars = (
        FEATURES_STAGE4["categorical"] if stage == 4 else FEATURES_STAGE8["categorical"]
    )
    cat_vars = [
        col
        for col in cat_vars
        if col in df_stage.columns and not df_stage[col].isna().all()
    ]
    if not cat_vars:
        return

    for cat_col in cat_vars[:3]:
        avg_fuel = (
            df_stage.groupby(cat_col, as_index=False)["PredictedFuel"]
            .mean()
            .sort_values("PredictedFuel", ascending=False)
        )
        if len(avg_fuel) > 10:
            avg_fuel = avg_fuel.head(10)

        fig = px.bar(
            avg_fuel,
            x=cat_col,
            y="PredictedFuel",
            title=f"Average Predicted Fuel by {cat_col} (Stage {stage})",
            labels={"PredictedFuel": "Avg Predicted Fuel (L)"},
        )
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)


def plot_fuel_consumption_heatmap(df: pl.DataFrame):
    """
    Create interactive geospatial heatmap showing fuel consumption hotspots.
    Only shows top consumption points to improve performance.
    """
    df_pandas = df.to_pandas()

    # Filter valid coordinates
    df_geo = df_pandas[
        (df_pandas["Latitude"].notna())
        & (df_pandas["Longitude"].notna())
        & (df_pandas["Latitude"] != 0)
        & (df_pandas["Longitude"] != 0)
        & (df_pandas["PredictedFuel"] > 0)
    ].copy()

    if len(df_geo) == 0:
        st.warning("No hay datos geográficos válidos disponibles para el mapa de calor")
        return

    # Limit to top 200 points for performance (adjust as needed)
    df_geo_top = df_geo.nlargest(50, "PredictedFuel")

    # Calculate center
    center_lat = df_geo_top["Latitude"].mean()
    center_lon = df_geo_top["Longitude"].mean()

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
    )

    # Prepare heatmap data: [lat, lon, weight] - only top points
    heat_data = [
        [row["Latitude"], row["Longitude"], row["PredictedFuel"]]
        for _, row in df_geo_top.iterrows()
    ]

    # Add heatmap layer
    plugins.HeatMap(
        heat_data,
        min_opacity=0.4,
        max_zoom=18,
        radius=20,
        blur=25,
        gradient={0.0: "blue", 0.3: "lime", 0.5: "yellow", 0.7: "orange", 1.0: "red"},
    ).add_to(m)

    # Add top 10 consumption points as red markers
    top_10 = df_geo_top.head(10)

    for idx, row in top_10.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=10,
            popup=folium.Popup(
                f"<b>Fuel:</b> {row['PredictedFuel']:.2f}L<br>"
                f"<b>Stage:</b> {row['StageSequence']}<br>"
                f"<b>Dest:</b> {row['Destination']}<br>"
                f"<b>Distance:</b> {row['Distance']:.0f}m",
                max_width=200,
            ),
            color="darkred",
            fill=True,
            fillColor="red",
            fillOpacity=0.8,
            weight=2,
        ).add_to(m)

    # Display map
    st_folium(m, width=None, height=600)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total puntos geoespaciales", f"{len(df_geo):,}")
    with col2:
        st.metric("Puntos en el mapa de calor", f"{len(df_geo_top):,}")
    with col3:
        st.metric(
            "Consumo maximo encontrado:", f"{df_geo['PredictedFuel'].max():.2f} L"
        )
    with col4:
        avg_fuel = df_geo["PredictedFuel"].mean()
        st.metric("Consumo promedio:", f"{avg_fuel:.2f} L")


def show():
    st.title("Modelo de predicciones de consumo de combustible con XGBoost")

    with st.sidebar:
        st.header("Truck Selection")
        truck_id = st.selectbox("Select Truck ID:", options=TRUCK_IDS, index=0)

    # Load models
    with st.spinner(f"Cargando modelos para {truck_id} desde MLflow..."):
        model_stage4, model_stage8 = load_models_from_mlflow(truck_id)

    if model_stage4 is None or model_stage8 is None:
        st.error(
            "No se pudieron cargar los modelos. Verifique la conexión a MLflow y el registro de modelos."
        )
        st.stop()

    st.success(f"Modelos cargados exitosamente para {truck_id}")

    # Load data
    with st.spinner(f"Cargando datos operativos para {truck_id}..."):
        df = load_data_from_clickhouse(truck_id)

    if df is None or df.is_empty():
        st.error("No hay datos disponibles para el camión seleccionado.")
        st.stop()

    st.success(f"Cargando {len(df):,} registros operativos")

    # Generate predictions
    with st.spinner("Generando predicciones..."):
        df_predictions = generate_predictions(df, model_stage4, model_stage8)

    # Display results
    st.header("Insights de Predicción")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Time Series",
            "Distribution & Summary",
            "Stage 4 (Empty Truck)",
            "Stage 8 (Loaded Truck)",
            "Geospatial Heatmap",
        ]
    )

    with tab1:
        st.subheader("Combustible esperado a lo largo del tiempo")
        plot_time_series_predictions_only(df_predictions)

    with tab2:
        st.subheader("Comportamiento general de las predicciones")
        plot_fuel_distribution_by_stage(df_predictions)
        show_prediction_summary(df_predictions)

    with tab3:
        st.subheader("Etapa 4: Relaciones clave")
        plot_prediction_vs_numeric(df_predictions, stage=4)
        plot_fuel_by_category(df_predictions, stage=4)

    with tab4:
        st.subheader("Etapa 8: Relaciones clave")
        plot_prediction_vs_numeric(df_predictions, stage=8)
        plot_fuel_by_category(df_predictions, stage=8)

    with tab5:
        st.subheader("Mapa de calor del consumo de combustible")
        st.info(
            "Red areas indicate higher fuel consumption. Red markers show top 10 consumption points."
        )
        plot_fuel_consumption_heatmap(df_predictions)

    # Data preview
    with st.expander("View Prediction Data"):
        display_cols = [
            "TimeStamp",
            "StageSequence",
            "PredictedFuel",
            "Distance",
            "SpeedAvg",
            "TotalMeasuredTonnage",
            "Destination",
            "Material",
            "Shovel",
            "Latitude",
            "Longitude",
            "Elevation",
        ]
        available_cols = [col for col in display_cols if col in df_predictions.columns]
        st.dataframe(
            df_predictions.select(available_cols).to_pandas(),
            height=400,
        )

    st.markdown("## 📊 Generación de reportes")
    add_report_generation_to_ui(df_predictions, truck_id)


if __name__ == "__main__":
    show()
