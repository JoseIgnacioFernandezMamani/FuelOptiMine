"""
Model Evaluation Dashboard

Evaluates trained XGBoost models by:
- Loading pre-trained models for selected truck
- Generating predictions on test data
- Displaying comprehensive metrics and visualizations
- Comparing Stage 4 vs Stage 8 performance
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import shap
from model.predictive.xgboost_model import XGBoostModel
from mlflow_server.config import (
    TRUCK_IDS,
    NUMERIC_PREDICTOR_VARS,
    CATEGORICAL_VARS,
)
import matplotlib.pyplot as plt


@st.cache_data
def train_and_evaluate_model(truck_id: str):
    """
    Train XGBoost model and return predictions with metrics.

    Args:
        truck_id: Truck identifier

    Returns:
        Tuple of (model, results, predictions_df)
    """
    model = XGBoostModel(
        truck_id=truck_id,
        numeric_predictor_vars=NUMERIC_PREDICTOR_VARS,
        categorical_vars=CATEGORICAL_VARS,
        max_cat_to_onehot=4,
    )

    # Load and transform data
    model.load_data()
    model.transform_cycles_data()

    # Train models
    results = model.train()

    # Generate predictions
    predictions_df = model.get_predictions()

    return model, results, predictions_df


def plot_metrics_comparison(results: dict):
    """
    Create bar chart comparing Stage 4 vs Stage 8 metrics.
    """
    metrics_4 = results["stage4"]["metrics"]
    metrics_8 = results["stage8"]["metrics"]

    metrics_to_plot = ["R2", "MAE", "RMSE", "MAPE_Safe"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=metrics_to_plot,
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2 + 1
        col = idx % 2 + 1

        fig.add_trace(
            go.Bar(
                x=["Stage 4", "Stage 8"],
                y=[metrics_4[metric], metrics_8[metric]],
                marker_color=["blue", "red"],
                text=[f"{metrics_4[metric]:.4f}", f"{metrics_8[metric]:.4f}"],
                textposition="outside",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text="Performance Metrics Comparison", height=600, showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_predictions_scatter(model, stage: int):
    """
    Create scatter plot of predictions vs actual for a specific stage.
    """
    if stage not in [4, 8]:
        st.error("Invalid stage. Must be 4 or 8.")
        return

    stage_key = "stage4" if stage == 4 else "stage8"

    if stage_key not in model.test_data:
        st.warning(f"No test data available for Stage {stage}")
        return

    data = model.test_data[stage_key]
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    residuals = y_true - y_pred

    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(
                size=8,
                color=residuals,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Residual (L)"),
            ),
            text=[
                f"True: {t:.2f}L<br>Pred: {p:.2f}L<br>Error: {r:.2f}L"
                for t, p, r in zip(y_true, y_pred, residuals)
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Predictions",
        )
    )

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name="Perfect Prediction",
        )
    )

    stage_name = "Empty Truck" if stage == 4 else "Loaded Truck"

    fig.update_layout(
        title=f"Stage {stage} ({stage_name}) - Predictions vs Actual",
        xaxis_title="Actual Fuel Consumed (L)",
        yaxis_title="Predicted Fuel Consumed (L)",
        template="plotly_white",
        height=500,
    )

    # Add metrics annotation
    fig.add_annotation(
        text=f"R² = {r2:.4f}<br>MAE = {mae:.2f}L<br>RMSE = {rmse:.2f}L",
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.95,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_residuals_analysis(model, stage: int):
    """
    Create residuals analysis plots.
    """
    stage_key = "stage4" if stage == 4 else "stage8"

    if stage_key not in model.test_data:
        return

    data = model.test_data[stage_key]
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    residuals = y_true - y_pred

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residuals Distribution", "Residuals vs Predicted"),
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color="lightblue",
            marker_line_color="black",
            marker_line_width=1,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(size=6, color="blue", opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_xaxes(title_text="Residual (L)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Fuel (L)", row=1, col=2)
    fig.update_yaxes(title_text="Residual (L)", row=1, col=2)

    stage_name = "Stage 4" if stage == 4 else "Stage 8"
    fig.update_layout(
        title_text=f"{stage_name} - Residuals Analysis",
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(model, stage: int):
    """
    Plot feature importance for a specific stage (all features).
    """
    importance_dict = model.get_feature_importance(stage=f"stage{stage}")
    stage_key = f"stage{stage}"

    if stage_key not in importance_dict:
        st.warning(f"No importance data for Stage {stage}")
        return

    df_importance = importance_dict[stage_key]  # All features, not limited

    fig = go.Figure(
        go.Bar(
            x=df_importance["importance"],
            y=df_importance["feature"],
            orientation="h",
            marker_color="teal",
            text=df_importance["importance"].round(4),
            textposition="outside",
        )
    )

    stage_name = "Stage 4 (Empty)" if stage == 4 else "Stage 8 (Loaded)"
    fig.update_layout(
        title=f"Feature Importance - {stage_name}",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, len(df_importance) * 25),  # Dynamic height
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_shap_summary(model, stage: int):
    """
    Generate SHAP summary plot using streamlit matplotlib rendering.
    """
    stage_key = "stage4" if stage == 4 else "stage8"
    stage_model = model.model_stage4 if stage == 4 else model.model_stage8

    if stage_key not in model.test_data:
        st.warning(f"No test data for Stage {stage}")
        return

    X_test = model.test_data[stage_key]["X"]

    try:
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(
                stage_model, feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(X_test, check_additivity=False)

        stage_name = "Stage 4 (Empty)" if stage == 4 else "Stage 8 (Loaded)"

        # SHAP Summary Plot
        st.subheader(f"SHAP Summary - {stage_name}")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test, feature_names=list(X_test.columns), show=False
        )
        st.pyplot(fig)

        # SHAP Feature Importance
        st.subheader(f"SHAP Feature Importance - {stage_name}")
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": list(X_test.columns), "mean_abs_shap": shap_importance}
        ).sort_values("mean_abs_shap", ascending=False)

        fig2 = go.Figure(
            go.Bar(
                x=importance_df["mean_abs_shap"],
                y=importance_df["feature"],
                orientation="h",
                marker_color="purple",
                text=importance_df["mean_abs_shap"].round(4),
                textposition="outside",
            )
        )

        fig2.update_layout(
            title=f"SHAP Mean Absolute Values - {stage_name}",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(400, len(importance_df) * 25),
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating SHAP: {str(e)}")


def plot_time_series_predictions(predictions_df: pl.DataFrame):
    """
    Plot time series of predictions colored by stage.
    """
    df_pandas = predictions_df.sort("TimeStampIni").to_pandas()

    fig = go.Figure()

    # Stage 4
    df_stage4 = df_pandas[df_pandas["StageSequence"] == 4]
    if len(df_stage4) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_stage4["TimeStampIni"],
                y=df_stage4["PredictedFuelXGBoost"],
                mode="markers",
                name="Stage 4 (Empty)",
                marker=dict(size=4, color="blue", opacity=0.6),
            )
        )

    # Stage 8
    df_stage8 = df_pandas[df_pandas["StageSequence"] == 8]
    if len(df_stage8) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_stage8["TimeStampIni"],
                y=df_stage8["PredictedFuelXGBoost"],
                mode="markers",
                name="Stage 8 (Loaded)",
                marker=dict(size=4, color="red", opacity=0.6),
            )
        )

    fig.update_layout(
        title="Predicted Fuel Consumption Over Time",
        xaxis_title="Date",
        yaxis_title="Predicted Fuel (L)",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def display_metrics_table(results: dict):
    """
    Display metrics in a formatted table.
    """
    metrics_4 = results["stage4"]["metrics"]
    metrics_8 = results["stage8"]["metrics"]

    metrics_data = {
        "Metric": ["R²", "MAE (L)", "RMSE (L)", "MAPE (%)", "Median AE (L)", "RMSLE"],
        "Stage 4": [
            f"{metrics_4['R2']:.4f}",
            f"{metrics_4['MAE']:.2f}",
            f"{metrics_4['RMSE']:.2f}",
            f"{metrics_4['MAPE_Safe']:.2f}",
            f"{metrics_4['MedianAE']:.2f}",
            f"{metrics_4['RMSLE']:.4f}",
        ],
        "Stage 8": [
            f"{metrics_8['R2']:.4f}",
            f"{metrics_8['MAE']:.2f}",
            f"{metrics_8['RMSE']:.2f}",
            f"{metrics_8['MAPE_Safe']:.2f}",
            f"{metrics_8['MedianAE']:.2f}",
            f"{metrics_8['RMSLE']:.4f}",
        ],
    }

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)


def show():
    """Main Streamlit application for model evaluation."""

    # Initialize session state
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "current_truck" not in st.session_state:
        st.session_state.current_truck = None

    st.title("Model Performance Evaluation")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        truck_id = st.selectbox("Select Truck:", TRUCK_IDS, index=0)

        st.info("This process may take several minutes depending on data volume.")

        if st.button("Train & Evaluate", type="primary", use_container_width=True):
            st.session_state.model_trained = True
            st.session_state.current_truck = truck_id
            st.rerun()

    # Check if user needs to train first
    if not st.session_state.model_trained:
        st.info("Click 'Train & Evaluate' in the sidebar to start model evaluation.")
        st.warning(
            "Note: Training process may take several minutes (5-15 min depending on data size)."
        )
        st.stop()

    # Reset if truck changed
    if st.session_state.current_truck != truck_id:
        st.session_state.model_trained = False
        st.info("Truck changed. Please click 'Train & Evaluate' again.")
        st.stop()

    # Train/load model
    with st.spinner(f"Training model for {truck_id}... This may take 5-15 minutes."):
        model, results, predictions_df = train_and_evaluate_model(truck_id)

    st.success(f"Model trained for {truck_id}")

    # Display overall metrics
    st.header("Overall Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Fuel", f"{results['total_consumed_fuel']:.2f} L")
    with col2:
        st.metric("Stage 4 Samples", f"{results['stage4']['samples']['test']}")
    with col3:
        st.metric("Stage 8 Samples", f"{results['stage8']['samples']['test']}")
    with col4:
        total_samples = (
            results["stage4"]["samples"]["test"] + results["stage8"]["samples"]["test"]
        )
        st.metric("Total Test Samples", f"{total_samples}")

    # Metrics comparison
    st.subheader("Metrics Comparison")
    display_metrics_table(results)
    plot_metrics_comparison(results)

    # Predictions scatter
    st.header("Predictions vs Actual")
    tab1, tab2 = st.tabs(["Stage 4 (Empty)", "Stage 8 (Loaded)"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            plot_predictions_scatter(model, stage=4)
        with col2:
            st.metric("R²", f"{results['stage4']['metrics']['R2']:.4f}")
            st.metric("MAE", f"{results['stage4']['metrics']['MAE']:.2f} L")
            st.metric("RMSE", f"{results['stage4']['metrics']['RMSE']:.2f} L")
            st.metric("MAPE", f"{results['stage4']['metrics']['MAPE_Safe']:.2f}%")

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            plot_predictions_scatter(model, stage=8)
        with col2:
            st.metric("R²", f"{results['stage8']['metrics']['R2']:.4f}")
            st.metric("MAE", f"{results['stage8']['metrics']['MAE']:.2f} L")
            st.metric("RMSE", f"{results['stage8']['metrics']['RMSE']:.2f} L")
            st.metric("MAPE", f"{results['stage8']['metrics']['MAPE_Safe']:.2f}%")

    # Residuals Analysis
    st.header("Residuals Analysis")
    tab1, tab2 = st.tabs(["Stage 4 (Empty)", "Stage 8 (Loaded)"])

    with tab1:
        plot_residuals_analysis(model, stage=4)

    with tab2:
        plot_residuals_analysis(model, stage=8)

    # Feature Importance
    st.header("Feature Importance Analysis")
    tab1, tab2 = st.tabs(["Stage 4 (Empty)", "Stage 8 (Loaded)"])

    with tab1:
        plot_feature_importance(model, stage=4)

    with tab2:
        plot_feature_importance(model, stage=8)

    # SHAP Analysis
    st.header("SHAP Explainability")
    tab1, tab2 = st.tabs(["Stage 4 (Empty)", "Stage 8 (Loaded)"])

    with tab1:
        plot_shap_summary(model, stage=4)

    with tab2:
        plot_shap_summary(model, stage=8)

    # Time series
    st.header("Predictions Over Time")
    plot_time_series_predictions(predictions_df)

    # Feature comparison
    st.header("Feature Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stage 4 Features")
        st.write(results["stage4"]["features"])
    with col2:
        st.subheader("Stage 8 Features")
        st.write(results["stage8"]["features"])

    # Raw data preview
    with st.expander("View Predictions Data"):
        st.dataframe(
            predictions_df.select(
                [
                    "TimeStampIni",
                    "StageSequence",
                    "Equipment",
                    "PredictedFuelXGBoost",
                    "FuelConsumed",
                    "residual",
                    "Distance",
                    "SpeedAvg",
                    "Destination",
                ]
            ).to_pandas(),
            height=400,
        )


if __name__ == "__main__":
    show()
