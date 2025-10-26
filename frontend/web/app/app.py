import streamlit as st
from pathlib import Path
import sys
import os

# ========== CONFIGURACIÓN DE PATHS SIMPLIFICADA ==========
def setup_project_path():
    """Configure project path directly and quickly"""
    current_file = Path(__file__).resolve()
    # From frontend/web/app/app.py -> go up 4 levels to root
    project_root = current_file.parent.parent.parent.parent
    
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root

# Execute configuration (instant)
PROJECT_ROOT = setup_project_path()

# ========== CONFIGURACIÓN DE STREAMLIT ==========
st.set_page_config(
    page_title="Análisis de Combustible", 
    page_icon="⛽", 
    layout="wide"
)

# Logo (quick load, only path)
LOGO_PATH = PROJECT_ROOT / "frontend" / "web" / "app" / "images" / "logo.png"
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH))

# ========== LAZY IMPORTS - Only load when needed ==========
def lazy_import_config():
    """Import configuration (fast, no heavy libraries)"""
    from mlflow_server.config import (
        TRUCK_IDS,
        NUMERIC_PREDICTOR_VARS,
        CATEGORICAL_VARS,
        MLFLOW_TRACKING_URI
    )
    return TRUCK_IDS, NUMERIC_PREDICTOR_VARS, CATEGORICAL_VARS, MLFLOW_TRACKING_URI

def lazy_import_db_utils():
    """Import database utilities (will take ~15s first time)"""
    from etl_core.load.utils import create_client, CH_CONFIG
    return create_client, CH_CONFIG

def lazy_import_report_generator():
    """Import report generator"""
    from model.predictive.report_generator import ReportGenerator
    return ReportGenerator

def lazy_import_heavy_libs():
    """Import heavy libraries only when needed"""
    import polars as pl
    import mlflow
    import mlflow.xgboost
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from datetime import datetime
    
    return {
        'pl': pl,
        'mlflow': mlflow,
        'mlflow_xgboost': mlflow.xgboost,
        'go': go,
        'px': px,
        'make_subplots': make_subplots,
        'np': np,
        'pd': pd,
        'shap': shap,
        'plt': plt,
        'r2_score': r2_score,
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'datetime': datetime
    }

# ========== CARGAR CONFIGURACIÓN (RÁPIDO) ==========
with st.spinner("⚡ Cargando configuración..."):
    try:
        TRUCK_IDS, NUMERIC_PREDICTOR_VARS, CATEGORICAL_VARS, MLFLOW_TRACKING_URI = lazy_import_config()
    except Exception as e:
        st.error(f"❌ Error cargando configuración: {e}")
        st.stop()

# ========== FEATURES DEFINITION ==========
FEATURES_STAGE4 = {
    "numeric": ["SpeedAvg", "Distance", "CycleDurationSeconds", "TimeEfficiencyPercentage"],
    "categorical": ["Shovel", "Destination", "DestinationType"]
}

FEATURES_STAGE8 = {
    "numeric": ["SpeedAvg", "TotalMeasuredTonnage", "Distance", "CycleDurationSeconds", "TimeEfficiencyPercentage"],
    "categorical": ["Destination", "DestinationType", "Material"]
}

# ========== MODEL FUNCTIONS WITH LAZY LOADING ==========
@st.cache_resource
def load_models(truck_id: str):
    """Load models from MLflow (lazy load mlflow)"""
    libs = lazy_import_heavy_libs()
    mlflow = libs['mlflow']
    mlflow_xgboost = libs['mlflow_xgboost']
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        m4 = mlflow_xgboost.load_model(f"models:/{truck_id}_stage4_fuel/Production")
        m8 = mlflow_xgboost.load_model(f"models:/{truck_id}_stage8_fuel/Production")
        return m4, m8
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None


@st.cache_data
def load_data(truck_id: str):
    """Load data from ClickHouse (lazy load DB and polars)"""
    libs = lazy_import_heavy_libs()
    pl = libs['pl']
    
    create_client, CH_CONFIG = lazy_import_db_utils()
    
    try:
        client = create_client(CH_CONFIG)
        query = f"""
        SELECT StageSequence, SpeedAvg, Distance, TimeEfficiencyPercentage, 
               MeasuredTonnage, TimeStampIni, TimeStampFin, FuelLevelLiters,
               Shovel, Destination, DestinationType, Material, LoadingZone
        FROM fuel_optimine.xgboost_fuel
        WHERE Equipment = '{truck_id}'
        ORDER BY SortTimestamp
        """
        df = pl.from_pandas(client.query_df(query))
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None


def get_metrics(truck_id: str):
    """Get model metrics from MLflow"""
    libs = lazy_import_heavy_libs()
    mlflow = libs['mlflow']
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        exp = mlflow.get_experiment_by_name(f"{truck_id}_fuel_prediction")
        if not exp:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs.empty:
            return None
        
        r = runs.iloc[0]
        metrics = {
            'stage4': {
                'metrics': {
                    'R2': r.get('metrics.stage4_R2', 0),
                    'MAE': r.get('metrics.stage4_MAE', 0),
                    'RMSE': r.get('metrics.stage4_RMSE', 0),
                    'MAPE_Safe': r.get('metrics.stage4_MAPE', 0),
                    'MedianAE': 0,
                    'RMSLE': r.get('metrics.stage4_RMSLE', 0),
                    'ExplainedVar': 0
                },
                'samples': {
                    'train': int(r.get('params.train_samples_stage4', 0)),
                    'test': int(r.get('params.test_samples_stage4', 0))
                }
            },
            'stage8': {
                'metrics': {
                    'R2': r.get('metrics.stage8_R2', 0),
                    'MAE': r.get('metrics.stage8_MAE', 0),
                    'RMSE': r.get('metrics.stage8_RMSE', 0),
                    'MAPE_Safe': r.get('metrics.stage8_MAPE', 0),
                    'MedianAE': 0,
                    'RMSLE': r.get('metrics.stage8_RMSLE', 0),
                    'ExplainedVar': 0
                },
                'samples': {
                    'train': int(r.get('params.train_samples_stage8', 0)),
                    'test': int(r.get('params.test_samples_stage8', 0))
                }
            }
        }
        return metrics
    except Exception as e:
        st.warning(f"No se pudieron obtener métricas: {str(e)}")
        return None


# ========== DATA TRANSFORMATION ==========
def transform_cycles(df):
    """Transform data into complete cycles"""
    libs = lazy_import_heavy_libs()
    pl = libs['pl']
    
    df = df.with_columns([
        pl.col("FuelLevelLiters").alias("MedianFuelLevelLiters"),
        pl.coalesce([pl.col("LoadingZone"), pl.col("Destination")]).alias("Destination")
    ])
    
    df = df.with_columns([
        pl.when((pl.col("StageSequence") == 4) | (pl.col("StageSequence") == 8) | (pl.col("StageSequence") == 1))
          .then(True).otherwise(False).alias("cycle_end")
    ])
    
    df = df.with_columns([
        pl.col("cycle_end").shift(1, fill_value=False).cum_sum().alias("cycle_group")
    ])
    
    result = df.group_by("cycle_group").agg([
        pl.col("StageSequence").sum().alias("StageSum"),
        pl.col("TimeStampIni").last().alias("TimeStampIni"),
        pl.col("TimeStampFin").last().alias("TimeStampFin"),
        (pl.col("SpeedAvg").filter(pl.col("SpeedAvg") > 5).median()
         .fill_null(pl.col("SpeedAvg").median())).alias("SpeedAvg"),
        pl.col("TimeEfficiencyPercentage").sum().alias("TimeEfficiencyPercentage"),
        pl.col("StageSequence").last().alias("StageSequence"),
        pl.col("Destination").last().alias("Destination"),
        pl.col("DestinationType").last().alias("DestinationType"),
        pl.col("Material").last().alias("Material"),
        pl.col("Shovel").last().alias("Shovel"),
        pl.col("MeasuredTonnage").sum().alias("TotalMeasuredTonnage"),
        pl.col("Distance").sum().alias("Distance"),
    ]).sort("cycle_group")
    
    result = result.filter(
        ((pl.col("StageSum") == 9) & (pl.col("StageSequence") == 4)) | 
        ((pl.col("StageSum") == 26) & (pl.col("StageSequence") == 8))
    ).drop("StageSum")
    
    result = result.with_columns([
        pl.when((pl.col("StageSequence") == 4) & (pl.col("StageSequence").shift(1) == 1))
          .then(pl.col("TimeStampIni").shift(1))
          .when((pl.col("StageSequence") == 8) & (pl.col("StageSequence").shift(1) == 4))
          .then(pl.col("TimeStampFin").shift(1))
          .otherwise(pl.col("TimeStampIni")).alias("TimeStampIni"),
        pl.when((pl.col("StageSequence") == 1) & (pl.col("StageSequence").shift(1) == 8))
          .then(pl.col("TimeStampIni"))
          .otherwise(pl.col("TimeStampFin")).alias("TimeStampFin"),
        pl.when((pl.col("StageSequence") == 4) & (pl.col("StageSequence").shift(1) == 1) & 
                (pl.col("TimeEfficiencyPercentage").shift(1) > 0))
          .then(pl.col("TimeEfficiencyPercentage").shift(1) + pl.col("TimeEfficiencyPercentage"))
          .otherwise(pl.col("TimeEfficiencyPercentage")).alias("TimeEfficiencyPercentage"),
    ])
    
    result = result.with_columns([
        pl.when(
            (((pl.col("TimeStampFin") - pl.col("TimeStampIni")).dt.total_seconds().abs() > 3600) | 
             ((pl.col("TimeStampFin") - pl.col("TimeStampIni")).dt.total_seconds().abs() < 50)) &
            (pl.col("SpeedAvg") > 0.1) & (pl.col("Distance") > 50) & (pl.col("Distance") <= 3600)
        )
        .then(
            pl.min_horizontal([
                (pl.col("Distance") / (pl.col("SpeedAvg") / 3.6)),
                pl.lit(float(900))
            ]).clip(lower_bound=180)
        )
        .otherwise((pl.col("TimeStampFin") - pl.col("TimeStampIni")).dt.total_seconds().abs())
        .alias("CycleDurationSeconds"),
        
        pl.when(pl.col("StageSequence") == 1).then(0)
          .otherwise(pl.col("TimeEfficiencyPercentage")).alias("TimeEfficiencyPercentage"),
        pl.when(pl.col("Destination").str.strip_chars().str.len_bytes() > 2)
          .then(pl.col("Destination")).otherwise(pl.lit("UNKNOWN")).alias("Destination"),
        pl.col("DestinationType").fill_null(pl.lit("LoadingZone")).alias("DestinationType"),
        pl.col("Material").fill_null(pl.lit("Empty")).alias("Material"),
        pl.col("Shovel").fill_null(pl.lit("Unknown")).alias("Shovel"),
    ])
    
    result = result.filter(
        (pl.col("StageSequence").is_in([4, 8])) & 
        (pl.col("StageSequence").is_not_null())
    )
    
    return result


def predict(df, m4, m8):
    """Generate predictions"""
    libs = lazy_import_heavy_libs()
    np = libs['np']
    pd = libs['pd']
    pl = libs['pl']
    
    df_pd = df.to_pandas()
    preds = np.zeros(len(df_pd))
    
    # Stage 4
    mask_4 = df_pd["StageSequence"] == 4
    if mask_4.any():
        X = df_pd.loc[mask_4, FEATURES_STAGE4["numeric"] + FEATURES_STAGE4["categorical"]].copy()
        for c in FEATURES_STAGE4["categorical"]:
            if c in X.columns:
                X[c] = X[c].astype("category")
                if len(X[c].cat.categories) == 0:
                    X[c] = pd.Categorical(['Unknown'] * len(X))
        try:
            preds[mask_4] = m4.predict(X)
        except Exception as e:
            st.warning(f"Error en Stage 4: {str(e)}")
            preds[mask_4] = 0
    
    # Stage 8
    mask_8 = df_pd["StageSequence"] == 8
    if mask_8.any():
        X = df_pd.loc[mask_8, FEATURES_STAGE8["numeric"] + FEATURES_STAGE8["categorical"]].copy()
        for c in FEATURES_STAGE8["categorical"]:
            if c in X.columns:
                valid = X[c].notna() & (X[c] != '')
                if not valid.all():
                    X.loc[~valid, c] = 'Unknown'
                X[c] = X[c].astype("category")
                if len(X[c].cat.categories) == 0:
                    X[c] = pd.Categorical(['Unknown'] * len(X))
        try:
            preds[mask_8] = m8.predict(X)
        except Exception as e:
            st.warning(f"Error en Stage 8: {str(e)}")
            preds[mask_8] = 0
    
    return df.with_columns([pl.Series("PredictedFuel", preds)])

def plot_distribution(df):
    """Distribution by stage"""
    libs = lazy_import_heavy_libs()
    px = libs['px']
    
    df_pd = df.to_pandas()
    fig = px.box(
        df_pd,
        x="StageSequence",
        y="PredictedFuel",
        color="StageSequence",
        labels={"StageSequence": "Etapa", "PredictedFuel": "Combustible (L)"},
        title="Distribución del Consumo de Combustible por Etapa"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def plot_metrics_comparison(metrics: dict):
    """Compare Stage 4 vs Stage 8 metrics"""
    libs = lazy_import_heavy_libs()
    go = libs['go']
    make_subplots = libs['make_subplots']
    
    if not metrics:
        st.warning("No hay métricas disponibles")
        return
    
    metrics_4 = metrics["stage4"]["metrics"]
    metrics_8 = metrics["stage8"]["metrics"]
    
    metrics_to_plot = ["R2", "MAE", "RMSE", "MAPE_Safe"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics_to_plot,
        specs=[[{"type": "bar"}, {"type": "bar"}], 
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=["Etapa 4", "Etapa 8"],
                y=[metrics_4[metric], metrics_8[metric]],
                marker_color=["blue", "red"],
                text=[f"{metrics_4[metric]:.4f}", f"{metrics_8[metric]:.4f}"],
                textposition="outside",
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Comparación de Métricas",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_feature_by_stage(df, feature: str, stage: int):
    """Plot feature vs predicted fuel for specific stage"""
    libs = lazy_import_heavy_libs()
    px = libs['px']
    pl = libs['pl']
    
    df_stage = df.filter(pl.col("StageSequence") == stage).to_pandas()
    
    if df_stage.empty or feature not in df_stage.columns:
        return
    
    fig = px.scatter(
        df_stage,
        x=feature,
        y="PredictedFuel",
        trendline="ols",
        title=f"Etapa {stage}: {feature} vs Combustible Predicho",
        labels={"PredictedFuel": "Combustible Predicho (L)"}
    )
    fig.update_layout(height=300, template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def show_summary(df):
    """Statistical summary"""
    libs = lazy_import_heavy_libs()
    pl = libs['pl']
    
    summary = (
        df.group_by("StageSequence")
        .agg([
            pl.col("PredictedFuel").mean().alias("Media"),
            pl.col("PredictedFuel").std().alias("Desv_Est"),
            pl.col("PredictedFuel").min().alias("Mínimo"),
            pl.col("PredictedFuel").max().alias("Máximo"),
            pl.len().alias("Ciclos")
        ])
        .sort("StageSequence")
        .to_pandas()
    )
    summary["Etapa"] = summary["StageSequence"].map({4: "Vacío", 8: "Cargado"})
    summary = summary[["Etapa", "Ciclos", "Media", "Desv_Est", "Mínimo", "Máximo"]]
    
    st.dataframe(summary.round(2), width='stretch')

def plot_predictions_scatter(model, stage: int):
    """Create scatter plot of predictions vs actual for a specific stage."""
    libs = lazy_import_heavy_libs()
    go = libs['go']
    np = libs['np']
    r2_score = libs['r2_score']
    mae = libs['mae']
    mse = libs['mse']
    
    if stage not in [4, 8]:
        st.error("Etapa inválida. Debe ser 4 u 8.")
        return

    stage_key = "stage4" if stage == 4 else "stage8"

    if stage_key not in model.test_data:
        st.warning(f"No hay datos de prueba disponibles para la Etapa {stage}")
        return

    data = model.test_data[stage_key]
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    residuals = y_true - y_pred

    r2 = r2_score(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))

    fig = go.Figure()

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
                f"Real: {t:.2f}L<br>Pred: {p:.2f}L<br>Error: {r:.2f}L"
                for t, p, r in zip(y_true, y_pred, residuals)
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Predicciones",
        )
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name="Predicción Perfecta",
        )
    )

    stage_name = "Camión Vacío" if stage == 4 else "Camión Cargado"

    fig.update_layout(
        title=f"Etapa {stage} ({stage_name}) - Predicciones vs Real",
        xaxis_title="Combustible Real Consumido (L)",
        yaxis_title="Combustible Predicho (L)",
        template="plotly_white",
        height=500,
    )

    fig.add_annotation(
        text=f"R² = {r2:.4f}<br>MAE = {mae_val:.2f}L<br>RMSE = {rmse:.2f}L",
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.95,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    st.plotly_chart(fig, width='stretch')

def plot_shap_summary(model, stage: int):
    """Generate SHAP summary plot with proper categorical handling"""
    libs = lazy_import_heavy_libs()
    shap = libs['shap']
    plt = libs['plt']
    np = libs['np']
    pd = libs['pd']
    go = libs['go']
    
    stage_key = "stage4" if stage == 4 else "stage8"
    stage_model = model.model_stage4 if stage == 4 else model.model_stage8

    if stage_key not in model.test_data:
        st.warning(f"No hay datos de prueba para la Etapa {stage}")
        return

    X_test = model.test_data[stage_key]["X"]

    try:
        with st.spinner("Calculando valores SHAP..."):
            # Convert categorical columns to numeric codes for SHAP
            X_test_numeric = X_test.copy()
            categorical_cols = X_test_numeric.select_dtypes(include=['category']).columns
            
            for col in categorical_cols:
                X_test_numeric[col] = X_test_numeric[col].cat.codes
            
            # Ensure all data is numeric and not string
            X_test_numeric = X_test_numeric.apply(pd.to_numeric, errors='coerce')
            
            # Fill any NaN values that might have appeared
            X_test_numeric = X_test_numeric.fillna(0)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(
                stage_model, 
                feature_perturbation="interventional"
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_numeric, check_additivity=False)

        stage_name = "Etapa 4 (Vacío)" if stage == 4 else "Etapa 8 (Cargado)"

        # SHAP Summary Plot
        st.subheader(f"Resumen SHAP - {stage_name}")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_test_numeric, 
            feature_names=list(X_test.columns), 
            show=False
        )
        plt.title(f"Importancia de Características - {stage_name}", fontsize=14, pad=20)
        st.pyplot(fig)

        # SHAP Feature Importance Bar Chart
        st.subheader(f"Importancia de Características SHAP - {stage_name}")
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
            title=f"Valores Absolutos Medios de SHAP - {stage_name}",
            xaxis_title="Media |Valor SHAP|",
            yaxis_title="Característica",
            template="plotly_white",
            height=max(400, len(importance_df) * 25),
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig2, width='stretch')

    except Exception as e:
        st.error(f"Error calculando SHAP: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

@st.cache_data
def train_and_evaluate_model(truck_id: str):
    """Train XGBoost model and return predictions with metrics."""
    from model.predictive.xgboost_model import XGBoostModel
    
    model = XGBoostModel(
        truck_id=truck_id,
        numeric_predictor_vars=NUMERIC_PREDICTOR_VARS,
        categorical_vars=CATEGORICAL_VARS,
        max_cat_to_onehot=4,
    )

    model.load_data()
    model.transform_cycles_data()
    results = model.train()
    predictions_df = model.get_predictions()

    return model, results, predictions_df
    
def add_report_generation_to_ui(df_predictions, truck_id: str, metrics: dict = None):
    """Generate and download PDF report"""
    libs = lazy_import_heavy_libs()
    datetime = libs['datetime']
    ReportGenerator = lazy_import_report_generator()
    
    st.header("📄 Generar Reporte Completo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros", f"{len(df_predictions):,}")
    with col2:
        stage4_count = (df_predictions["StageSequence"] == 4).sum()
        st.metric("Ciclos Vacíos", f"{stage4_count:,}")
    with col3:
        stage8_count = (df_predictions["StageSequence"] == 8).sum()
        st.metric("Ciclos Cargados", f"{stage8_count:,}")
    
    st.markdown("---")
    
    if st.button("🎯 Generar Reporte PDF", type="primary", width='stretch'):
        with st.spinner("Generando reporte completo..."):
            try:
                pdf_bytes = ReportGenerator(
                    df=df_predictions,
                    truck_id=truck_id,
                    model_metrics=metrics
                ).generate_pdf_report()
                
                st.success("✅ Reporte generado exitosamente!")
                
                filename = f"Reporte_Combustible_{truck_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📥 Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    width='stretch'
                )
            except Exception as e:
                st.error(f"Error generando reporte: {str(e)}")
                st.exception(e)

# ========== MAIN APPLICATION ==========
st.title("🏭 Predicción de Consumo de Combustible")

with st.sidebar:
    st.header("⚙️ Configuración")
    truck_id = st.selectbox("Seleccionar Camión", TRUCK_IDS, index=0)

    if st.button("🚀 Cargar & Predecir", type="primary", width='stretch'):
        st.session_state.ready = True
        st.session_state.truck = truck_id
        if 'last_truck' in st.session_state and st.session_state.last_truck != truck_id:
            st.cache_data.clear()
        st.session_state.last_truck = truck_id

if "ready" not in st.session_state:
    st.info("👈 Selecciona un camión y haz clic en **Cargar & Predecir**")
    st.stop()

# Load and process (AQUÍ se cargan las librerías pesadas solo cuando el usuario hace click)
with st.spinner("⚡ Cargando modelos..."):
    m4, m8 = load_models(st.session_state.truck)

if not m4 or not m8:
    st.error("No se pudieron cargar los modelos")
    st.stop()

with st.spinner("Obteniendo métricas del modelo..."):
    metrics = get_metrics(st.session_state.truck)

with st.spinner("Cargando datos de la base de datos..."):
    df = load_data(st.session_state.truck)

if df is None or df.is_empty():
    st.error("No hay datos disponibles")
    st.stop()

with st.spinner("Procesando ciclos..."):
    df_cycles = transform_cycles(df)

if df_cycles is None or df_cycles.is_empty():
    st.error("No se pudieron procesar ciclos válidos")
    st.stop()

with st.spinner("Generando predicciones..."):
    df_pred = predict(df_cycles, m4, m8)

st.success(f"✅ {len(df_pred):,} ciclos procesados ​​con éxito")

# Visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Evaluación del Modelo",
    "📊 Distribución",
    "📉 Métricas",
    "🔍 Análisis de Características",
    "📄 Informe PDF"
])


with tab1:
    st.header("Entrenamiento y evaluación de modelos")
    st.write("Realizará el entrenamiento en tiempo real del modelo XGBoost para el camión seleccionado y mostrará las métricas de rendimiento junto con gráficos de predicciones y análisis SHAP, con los datos disponibles en la base de datos.")
    
    if st.button("🚀 Entrenar y evaluar el modelo", type="primary", width='stretch'):
        with st.spinner("Modelo de entrenamiento en tiempo real..."):
            model, results, predictions_df = train_and_evaluate_model(st.session_state.truck)
            
        if model and results:
            st.success("✅ Modelo entrenada con éxito!")
            
            # Mostrar métricas para ambos stages
            st.subheader("📊 Camión vacío")
            col1, col2, col3 = st.columns(3)
            if "stage4" in results:
                metrics_res = results["stage4"]
                with col1:
                    st.metric("R² Score", f"{metrics_res.get('r2', 0):.4f}")
                with col2:
                    st.metric("MAE", f"{metrics_res.get('mae', 0):.2f} L")
                with col3:
                    st.metric("RMSE", f"{metrics_res.get('rmse', 0):.2f} L")
            
            plot_predictions_scatter(model, 4)
            plot_shap_summary(model, 4)
            
            st.markdown("---")
            
            # Stage 8
            st.subheader("📊 Camión con carga")
            col1, col2, col3 = st.columns(3)
            if "stage8" in results:
                metrics_res = results["stage8"]
                with col1:
                    st.metric("R² Score", f"{metrics_res.get('r2', 0):.4f}")
                with col2:
                    st.metric("MAE", f"{metrics_res.get('mae', 0):.2f} L")
                with col3:
                    st.metric("RMSE", f"{metrics_res.get('rmse', 0):.2f} L")
            
            plot_predictions_scatter(model, 8)
            plot_shap_summary(model, 8)

with tab2:
    plot_distribution(df_pred)
    st.subheader("Resumen Estadístico")
    show_summary(df_pred)

with tab3:
    if metrics:
        st.subheader("Métricas de rendimiento del modelo")
        plot_metrics_comparison(metrics)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stage 4 R²", f"{metrics['stage4']['metrics']['R2']:.4f}")
            st.metric("Stage 4 MAE", f"{metrics['stage4']['metrics']['MAE']:.2f} L")
        with col2:
            st.metric("Stage 8 R²", f"{metrics['stage8']['metrics']['R2']:.4f}")
            st.metric("Stage 8 MAE", f"{metrics['stage8']['metrics']['MAE']:.2f} L")
    else:
        st.warning("No hay métricas disponibles de MLflow")

with tab4:
    st.subheader("Características y su impacto en el consumo de combustible")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Camión vacío**")
        plot_feature_by_stage(df_pred, "Distance", 4)
        plot_feature_by_stage(df_pred, "SpeedAvg", 4)
    
    with col2:
        st.write("**Camión con carga**")
        plot_feature_by_stage(df_pred, "Distance", 8)
        plot_feature_by_stage(df_pred, "TotalMeasuredTonnage", 8)

with tab5:
    add_report_generation_to_ui(df_pred, st.session_state.truck, metrics)

# Data preview
with st.expander("🔍 Ver datos de predicción"):
    cols = ["TimeStampIni", "StageSequence", "PredictedFuel", "Distance",
            "SpeedAvg", "TotalMeasuredTonnage", "Destination", "Material"]
    available_cols = [c for c in cols if c in df_pred.columns]
    st.dataframe(df_pred.select(available_cols).to_pandas(), height=400)
