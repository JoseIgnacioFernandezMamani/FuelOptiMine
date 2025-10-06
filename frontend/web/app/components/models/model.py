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
from etl_core.load.utils import create_client, CH_CONFIG
from model.predictive.mlflow_config import TRUCK_IDS

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"


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

        print("exito")

        return model_stage4, model_stage8

    except Exception as e:
        st.error(f"Error loading models from MLflow: {str(e)}")
        st.info(
            "Make sure models are registered and promoted to 'Production' stage in MLflow"
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
            CycleId
        FROM fuel_optimine.xgboost_fuel
        WHERE Equipment = '{truck_id}'
          AND StageSequence IN (4, 8)
        ORDER BY SortTimestamp
        """

        data = client.query_df(query)

        df = pl.from_pandas(data)

        # Calculate CycleDurationSeconds  and PredictedFuel
        df = df.with_columns(
            [
                (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                .dt.total_seconds()
                .alias("CycleDurationSeconds"),
                pl.col("MeasuredTonnage").alias("TotalMeasuredTonnage"),
            ]
        )

        print(df.head(5))
        return df

    except Exception as e:
        st.error(f"Error loading data from ClickHouse: {str(e)}")
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

        # Convert categorical columns
        for cat_col in FEATURES_STAGE4["categorical"]:
            if cat_col in X_stage4.columns:
                X_stage4[cat_col] = X_stage4[cat_col].astype("category")

        predictions[mask_4] = model_stage4.predict(X_stage4)

    # Predict Stage 8 (loaded truck)
    mask_8 = df_pandas["StageSequence"] == 8
    if mask_8.any():
        features = FEATURES_STAGE8["numeric"] + FEATURES_STAGE8["categorical"]
        X_stage8 = df_pandas.loc[mask_8, features].copy()

        # Convert categorical columns
        for cat_col in FEATURES_STAGE8["categorical"]:
            if cat_col in X_stage8.columns:
                X_stage8[cat_col] = X_stage8[cat_col].astype("category")

        predictions[mask_8] = model_stage8.predict(X_stage8)

    # Add predictions to DataFrame
    result = df.with_columns(
        [
            pl.Series("PredictedFuel", predictions),
        ]
    )
    # Calculate
    return result


def plot_time_series_predictions_only(df: pl.DataFrame):
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

    # Colorear puntos por etapa
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
    st.subheader("📊 Prediction Summary by Stage")
    st.dataframe(summary.round(3), use_container_width=True)


def plot_prediction_vs_numeric(df: pl.DataFrame, stage: int):
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

    vars_to_plot = numeric_vars[:4]  # Limitar a 4 para claridad
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

    for cat_col in cat_vars[:3]:  # Máximo 3 categorías
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


# ───────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ───────────────────────────────────────────────


def show():
    st.title("⛽ XGBoost Fuel Consumption Predictions from MLflow")

    with st.sidebar:
        st.header("🚚 Truck Selection")
        truck_id = st.selectbox("Select Truck ID:", options=TRUCK_IDS, index=0)

    # Load models
    with st.spinner(f"Loading models for {truck_id} from MLflow..."):
        model_stage4, model_stage8 = load_models_from_mlflow(truck_id)

    if model_stage4 is None or model_stage8 is None:
        st.error(
            "Failed to load models. Check MLflow connection and model registration."
        )
        st.stop()

    st.success(f"✅ Models loaded successfully for {truck_id}")

    # Load data
    with st.spinner(f"Loading operational data for {truck_id}..."):
        df = load_data_from_clickhouse(truck_id)

    if df is None or df.is_empty():
        st.error("No data available for selected truck.")
        st.stop()

    st.success(f"✅ Loaded {len(df):,} operational records")

    # Generate predictions
    with st.spinner("Generating predictions..."):
        df_predictions = generate_predictions(df, model_stage4, model_stage8)

    # Display results
    st.header("📊 Prediction Insights")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Time Series",
            "📉 Distribution & Summary",
            "🔍 Stage 4 (Empty Truck)",
            "🔍 Stage 8 (Loaded Truck)",
        ]
    )

    with tab1:
        st.subheader("Predicted Fuel Over Time")
        plot_time_series_predictions_only(df_predictions)

    with tab2:
        st.subheader("Overall Prediction Behavior")
        plot_fuel_distribution_by_stage(df_predictions)
        show_prediction_summary(df_predictions)

    with tab3:
        st.subheader("Stage 4 – Key Relationships")
        plot_prediction_vs_numeric(df_predictions, stage=4)
        plot_fuel_by_category(df_predictions, stage=4)

    with tab4:
        st.subheader("Stage 8 – Key Relationships")
        plot_prediction_vs_numeric(df_predictions, stage=8)
        plot_fuel_by_category(df_predictions, stage=8)

    # Data preview
    with st.expander("📋 View Prediction Data"):
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
        ]
        available_cols = [col for col in display_cols if col in df_predictions.columns]
        st.dataframe(
            df_predictions.select(available_cols).to_pandas(),
            height=400,
        )


if __name__ == "__main__":
    show()
