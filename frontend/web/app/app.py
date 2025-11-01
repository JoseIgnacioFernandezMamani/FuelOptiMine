import streamlit as st
from pathlib import Path
import sys
import os

# ========== DELETE WARNINGS SOON ==========
import os
import logging
import warnings

# Ocultar warnings de Python
warnings.filterwarnings("ignore")

# Silenciar logs de Streamlit
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Silenciar logs de Plotly y deprecation internos
logging.getLogger('plotly').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# También silenciar mensajes que salen directo a la consola
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
    """Get model metrics from MLflow - FIXED VERSION"""
    libs = lazy_import_heavy_libs()
    mlflow = libs['mlflow']
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        exp = mlflow.get_experiment_by_name(f"{truck_id}_fuel_prediction")
        if not exp:
            st.warning(f"No se encontró experimento para {truck_id}")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            st.warning("No se encontraron runs de producción")
            return None
        
        r = runs.iloc[0]
        
        # Helper function to safely get metric values
        def get_metric(key, default=0.0):
            val = r.get(key, default)
            if val is None or val == '' or (isinstance(val, float) and val == 0.0):
                return default
            return float(val)
        
        def get_param(key, default=0):
            val = r.get(key, default)
            if val is None or val == '':
                return default
            try:
                return int(float(val))
            except:
                return default
        
        metrics = {
            'stage4': {
                'metrics': {
                    'R2': get_metric('metrics.stage4_R2'),
                    'MAE': get_metric('metrics.stage4_MAE'),
                    'RMSE': get_metric('metrics.stage4_RMSE'),
                    'MAPE_Safe': get_metric('metrics.stage4_MAPE'),
                    'RMSLE': get_metric('metrics.stage4_RMSLE')
                },
                'samples': {
                    'train': get_param('params.train_samples_stage4'),
                    'test': get_param('params.test_samples_stage4')
                }
            },
            'stage8': {
                'metrics': {
                    'R2': get_metric('metrics.stage8_R2'),
                    'MAE': get_metric('metrics.stage8_MAE'),
                    'RMSE': get_metric('metrics.stage8_RMSE'),
                    'MAPE_Safe': get_metric('metrics.stage8_MAPE'),
                    'RMSLE': get_metric('metrics.stage8_RMSLE')
                },
                'samples': {
                    'train': get_param('params.train_samples_stage8'),
                    'test': get_param('params.test_samples_stage8')
                }
            }
        }
        
        return metrics
    except Exception as e:
        st.warning(f"Error obteniendo métricas: {str(e)}")
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
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)

def plot_shap_analysis(truck_id: str):
    """
    Generate SHAP analysis for model interpretability
    """
    """
    Generate SHAP analysis for both stages with a single button
    """
    import shap
    import matplotlib.pyplot as plt

    libs = lazy_import_heavy_libs()
    pd = libs['pd']
    np = libs['np']
    
    try:
        # Cargar ambos modelos
        mlflow = libs['mlflow']
        mlflow_xgboost = libs['mlflow_xgboost']
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_4 = mlflow_xgboost.load_model(f"models:/{truck_id}_stage4_fuel/Production")
        model_8 = mlflow_xgboost.load_model(f"models:/{truck_id}_stage8_fuel/Production")
        
        # Cargar datos
        df = load_data(truck_id)
        if df is None or df.is_empty():
            st.warning("No hay datos disponibles para análisis SHAP")
            return
            
        df_cycles = transform_cycles(df)
        
        # Crear pestañas para cada etapa
        tab1, tab2 = st.tabs(["🚛 Camión sin carga (Etapa 4)", "🚛 Camión con carga (Etapa 8)"])
        
        with tab1:
            st.subheader("Análisis SHAP - Etapa 4")
            _plot_shap_for_stage(model_4, df_cycles, 4, truck_id)
        
        with tab2:
            st.subheader("Análisis SHAP - Etapa 8") 
            _plot_shap_for_stage(model_8, df_cycles, 8, truck_id)
            
    except Exception as e:
        st.error(f"Error en análisis SHAP: {str(e)}")

def _plot_shap_for_stage(model, df_cycles, stage: int, truck_id: str):
    """
    Helper function to plot SHAP for a specific stage
    """
    import shap
    import matplotlib.pyplot as plt
    libs = lazy_import_heavy_libs()
    pd = libs['pd']
    np = libs['np']
    pl = libs['pl']
    
    try:
        # Preparar características según la etapa
        if stage == 4:
            features = FEATURES_STAGE4["numeric"] + FEATURES_STAGE4["categorical"]
        else:
            features = FEATURES_STAGE8["numeric"] + FEATURES_STAGE8["categorical"]
        
        # Filtrar datos de la etapa
        df_stage = df_cycles.filter(pl.col("StageSequence") == stage)
        if df_stage.is_empty():
            st.warning(f"No hay datos para la etapa {stage}")
            return
            
        # Seleccionar y limpiar datos
        df_sample = df_stage.select(features).to_pandas()
        
        # Limpiar valores numéricos
        df_sample = clean_numeric_values(df_sample)
        
        # Preprocesar variables categóricas
        for col in df_sample.columns:
            if df_sample[col].dtype == 'object':
                # Convertir a códigos numéricos
                df_sample[col] = df_sample[col].astype('category').cat.codes
        
        # Tomar muestra para no sobrecargar
        if len(df_sample) > 100:
            df_sample = df_sample.sample(100, random_state=42)
        
        with st.spinner(f"Calculando valores SHAP para etapa {stage}..."):
            # Crear explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_sample)
            
            # 1. Gráfica de importancia de características
            st.markdown("**📊 Importancia de Características**")
            fig_importance, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, df_sample, plot_type="bar", show=False)
            ax.set_title(f"Importancia SHAP - Etapa {stage}")
            st.pyplot(fig_importance)
            plt.close(fig_importance)
            
            # 2. Gráfica summary plot
            st.markdown("**🔍 Impacto de Características**")
            fig_summary, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, df_sample, show=False)
            ax.set_title(f"Impacto SHAP - Etapa {stage}")
            st.pyplot(fig_summary)
            plt.close(fig_summary)
            
            # 3. Mostrar valores SHAP en tabla
            st.markdown("**📋 Valores SHAP Promedio**")
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'Característica': df_sample.columns,
                'Importancia_SHAP': feature_importance
            }).sort_values('Importancia_SHAP', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error en análisis SHAP para etapa {stage}: {str(e)}")

def clean_numeric_values(df_sample):
    """
    Limpia valores numéricos que están en formato string
    """
    import re
    import pandas as pd
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object':
            # Intenta convertir a numérico
            df_sample[col] = pd.to_numeric(df_sample[col], errors='ignore')
            
            # Si aún es object, intenta limpiar strings como '[1.4301796E1]'
            if df_sample[col].dtype == 'object':
                def clean_value(x):
                    if isinstance(x, str):
                        # Remover corchetes y espacios
                        cleaned = re.sub(r'[\[\]\s]', '', x)
                        try:
                            return float(cleaned)
                        except:
                            return x
                    return x
                
                df_sample[col] = df_sample[col].apply(clean_value)
    
    return df_sample

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
        
        val_4 = metrics_4[metric]
        val_8 = metrics_8[metric]
        
        # Format text based on metric type
        if metric == "MAE":
            text_4 = f"{val_4:.3f}"
            text_8 = f"{val_8:.3f}"
        else:
            text_4 = f"{val_4:.4f}"
            text_8 = f"{val_8:.4f}"
        
        fig.add_trace(
            go.Bar(
                x=["Etapa 4", "Etapa 8"],
                y=[val_4, val_8],
                marker_color=["blue", "red"],
                text=[text_4, text_8],
                textposition="outside",
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Comparación de Métricas del Modelo",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)


def plot_feature_by_stage(df, feature: str, stage: int):
    """Plot feature vs predicted fuel for specific stage - WITHOUT TRENDLINE"""
    libs = lazy_import_heavy_libs()
    px = libs['px']
    pl = libs['pl']
    
    df_stage = df.filter(pl.col("StageSequence") == stage).to_pandas()
    
    if df_stage.empty or feature not in df_stage.columns:
        return

    features_translation_dict = {
        "SpeedAvg": "Velocidad promedio",
        "Distance": "Distancia",
        "CycleDurationSeconds": "Duración del ciclo en segundos",
        "TimeEfficiencyPercentage": "Porcentaje de eficiencia de tiempo",
        "TotalMeasuredTonnage": "Tonelaje total medido",
        "Shovel": "Pala",
        "Destination": "Destino",
        "DestinationType": "Tipo de destino",
        "Material": "Material"
    }

    fig = px.scatter(
        df_stage,
        x=feature,
        y="PredictedFuel",
        title=f"{features_translation_dict[feature]} vs Combustible Predicho",
        labels={"PredictedFuel": "Combustible Predicho (L)"},
        opacity=0.6
    )
    fig.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)


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
        text=f"R² = {r2:.4f}<br>MAE = {mae_val:.3f}L<br>RMSE = {rmse:.2f}L",
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.95,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)


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

    if st.button("🎯 Generar Reporte PDF"):
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
                    
                )
            except Exception as e:
                st.error(f"Error generando reporte: {str(e)}")
                st.exception(e)


# ========== MAIN APPLICATION ==========
st.title("Predicción de Consumo de Combustible")

with st.sidebar:
    st.header("⚙️ Configuración")
    truck_id = st.selectbox("Seleccionar Camión", TRUCK_IDS, index=0)

    if st.button("🚀 Cargar & Predecir"):
        st.session_state.ready = True
        st.session_state.truck = truck_id
        if 'last_truck' in st.session_state and st.session_state.last_truck != truck_id:
            st.cache_data.clear()
        st.session_state.last_truck = truck_id

if "ready" not in st.session_state:
    st.info("👈 Selecciona un camión y haz clic en **Cargar & Predecir**")
    st.stop()

# Load and process
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

st.success(f"✅ {len(df_pred):,} ciclos procesados con éxito")

# ========== ONLY 2 TABS ==========
tab1, tab2 = st.tabs(["🤖 Evaluación del Modelo", "📄 Informe PDF"])

with tab1:
    st.header(f"📊 Evaluación del Modelo para el camión {truck_id}")
    
    # ========== SECCIÓN 1: MÉTRICAS DE PRODUCCIÓN ==========
    if metrics:
        st.subheader("📈 Métricas de Producción (MLflow)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚛 Camión sin carga**")
            m4_metrics = metrics['stage4']['metrics']
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("R² Score", f"{m4_metrics['R2']:.4f}")
            with subcol2:
                st.metric("MAE", f"{m4_metrics['MAE']:.3f} L")
            with subcol3:
                st.metric("RMSE", f"{m4_metrics['RMSE']:.2f} L")
        
        with col2:
            st.markdown("**🚛 Camión con carga**")
            m8_metrics = metrics['stage8']['metrics']
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("R² Score", f"{m8_metrics['R2']:.4f}")
            with subcol2:
                st.metric("MAE", f"{m8_metrics['MAE']:.3f} L")
            with subcol3:
                st.metric("RMSE", f"{m8_metrics['RMSE']:.2f} L")
        
        st.markdown("---")
        plot_metrics_comparison(metrics)
    else:
        st.info("ℹ️ No hay métricas de producción disponibles en MLflow")
    
    st.markdown("---")
    
    # ========== SECCIÓN 2: ENTRENAMIENTO Y EVALUACIÓN ==========
    st.subheader("🎯 Entrenamiento y Evaluación del Modelo")
    st.write("Entrena el modelo XGBoost en tiempo real con los datos actuales y visualiza las predicciones vs valores reales. Esto puede tardar unos segundos.")
    
    if st.button("🚀 Entrenar y Evaluar Modelo", type="primary"):
        with st.spinner("⚡ Entrenando modelo en tiempo real..."):
            model, results, predictions_df = train_and_evaluate_model(st.session_state.truck)
        
        if model and results:
            st.success("✅ Modelo entrenado con éxito!")
            
            # Stage 4 Results
            st.markdown("---")
            st.subheader("📊 Comparación de niveles reales (aprox.) y predicciones de combustible (modelo) durante ciclo de camion sin carga")
            plot_predictions_scatter(model, 4)
            
            # Stage 8 Results
            st.markdown("---")
            st.subheader("📊 Comparación de niveles reales (aprox.) y predicciones de combustible (modelo) durante ciclo de camion con carga")
            plot_predictions_scatter(model, 8)
            
        else:
            st.error("❌ Error durante el entrenamiento del modelo")
    
    st.markdown("---")
    
    # ========== SECCIÓN 3: DISTRIBUCIÓN Y ESTADÍSTICAS ==========
    st.subheader("📊 Distribución del Consumo de Combustible")
    plot_distribution(df_pred)
    
    st.subheader("📈 Resumen Estadístico")
    show_summary(df_pred)
    
    st.markdown("---")
    
    # ========== SECCIÓN 4: ANÁLISIS DE CARACTERÍSTICAS ==========
    st.subheader("🔍 Análisis de variables numericos vs Predicción de combustible")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚛 Camión sin carga**")
        for feat in FEATURES_STAGE4["numeric"]:
            if feat in df_pred.columns:
                plot_feature_by_stage(df_pred, feat, 4)
    
    with col2:
        st.markdown("**🚛 Camión con Carga**")
        for feat in FEATURES_STAGE8["numeric"]:
            if feat in df_pred.columns:
                plot_feature_by_stage(df_pred, feat, 8)
    
    st.markdown("---")

    # ========== SECCIÓN 5: VISTA PREVIA DE DATOS ==========
    with st.expander("🔍 Ver Datos de Predicción Completos"):
        cols = ["TimeStampIni", "TimeStampFin", "StageSequence", "PredictedFuel", 
                "Distance", "SpeedAvg", "CycleDurationSeconds", "TimeEfficiencyPercentage",
                "TotalMeasuredTonnage", "Destination", "DestinationType", "Material", "Shovel"]
        available_cols = [c for c in cols if c in df_pred.columns]
        st.dataframe(
            df_pred.select(available_cols).to_pandas(), 
            height=400, 
            width='stretch'
        )

with tab2:
    add_report_generation_to_ui(df_pred, st.session_state.truck, metrics)