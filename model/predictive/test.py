import polars as pl
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from datetime import datetime
from etl_core.load.utils import create_client, CH_CONFIG
from mlflow_server.config import MLFLOW_TRACKING_URI
from model.predictive.report_generator import ReportGenerator

TRUCK_ID = "T-210"
FEATURES_STAGE4 = {"numeric": ["SpeedAvg", "Distance", "CycleDurationSeconds", "TimeEfficiencyPercentage"], "categorical": ["Shovel", "Destination", "DestinationType"]}
FEATURES_STAGE8 = {"numeric": ["SpeedAvg", "TotalMeasuredTonnage", "Distance", "CycleDurationSeconds", "TimeEfficiencyPercentage"], "categorical": ["Destination", "DestinationType", "Material"]}

def load_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        print(f"Cargando modelos para {TRUCK_ID}...")
        m4 = mlflow.xgboost.load_model(f"models:/{TRUCK_ID}_stage4_fuel/Production")
        m8 = mlflow.xgboost.load_model(f"models:/{TRUCK_ID}_stage8_fuel/Production")
        print("✓ Modelos cargados")
        return m4, m8
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def load_data():
    try:
        print("Cargando datos...")
        client = create_client(CH_CONFIG)
        query = f"""
        SELECT StageSequence, SpeedAvg, Distance, TimeEfficiencyPercentage, 
               MeasuredTonnage, TimeStampIni, TimeStampFin, FuelLevelLiters,
               Shovel, Destination, DestinationType, Material, LoadingZone
        FROM fuel_optimine.xgboost_fuel
        WHERE Equipment = '{TRUCK_ID}'
        ORDER BY SortTimestamp
        """
        df = pl.from_pandas(client.query_df(query))
        print(f"✓ {len(df):,} registros cargados")
        return df
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def transform_cycles(df):
    print("Transformando datos en ciclos...")
    df = df.with_columns([
        pl.col("FuelLevelLiters").alias("MedianFuelLevelLiters"),
        pl.coalesce([pl.col("LoadingZone"), pl.col("Destination")]).alias(
                    "Destination"
                )
    ])
    
    df = df.with_columns([
        pl.when((pl.col("StageSequence") == 4) | (pl.col("StageSequence") == 8) | (pl.col("StageSequence") == 1)).then(True).otherwise(False).alias("cycle_end")
    ])
    
    df = df.with_columns([pl.col("cycle_end").shift(1, fill_value=False).cum_sum().alias("cycle_group")])
    
    result = df.group_by("cycle_group").agg([
        pl.col("StageSequence").sum().alias("StageSum"),
        pl.col("TimeStampIni").last().alias("TimeStampIni"),
        pl.col("TimeStampFin").last().alias("TimeStampFin"),
        (pl.col("SpeedAvg").filter(pl.col("SpeedAvg") > 5).median().fill_null(pl.col("SpeedAvg").median())).alias("SpeedAvg"),
        pl.col("TimeEfficiencyPercentage").sum().alias("TimeEfficiencyPercentage"),
        pl.col("StageSequence").last().alias("StageSequence"),
        pl.col("Destination").last().alias("Destination"),
        pl.col("DestinationType").last().alias("DestinationType"),
        pl.col("Material").last().alias("Material"),
        pl.col("Shovel").last().alias("Shovel"),
        pl.col("MeasuredTonnage").sum().alias("TotalMeasuredTonnage"),
        pl.col("Distance").sum().alias("Distance"),
    ]).sort("cycle_group")
    
    result = result.filter(((pl.col("StageSum") == 9) & (pl.col("StageSequence") == 4)) | ((pl.col("StageSum") == 26) & (pl.col("StageSequence") == 8))).drop("StageSum")
    
    result = result.with_columns([
        pl.when((pl.col("StageSequence") == 4) & (pl.col("StageSequence").shift(1) == 1)).then(pl.col("TimeStampIni").shift(1))
          .when((pl.col("StageSequence") == 8) & (pl.col("StageSequence").shift(1) == 4)).then(pl.col("TimeStampFin").shift(1))
          .otherwise(pl.col("TimeStampIni")).alias("TimeStampIni"),
        pl.when((pl.col("StageSequence") == 1) & (pl.col("StageSequence").shift(1) == 8)).then(pl.col("TimeStampIni"))
          .otherwise(pl.col("TimeStampFin")).alias("TimeStampFin"),
        pl.when((pl.col("StageSequence") == 4) & (pl.col("StageSequence").shift(1) == 1) & (pl.col("TimeEfficiencyPercentage").shift(1) > 0))
          .then(pl.col("TimeEfficiencyPercentage").shift(1) + pl.col("TimeEfficiencyPercentage"))
          .otherwise(pl.col("TimeEfficiencyPercentage")).alias("TimeEfficiencyPercentage"),
    ])
    
    result = result.with_columns([
        pl.when((((pl.col("TimeStampFin") - pl.col("TimeStampIni")).dt.total_seconds().abs() > 3600) | 
                 ((pl.col("TimeStampFin") - pl.col("TimeStampIni")).dt.total_seconds().abs() < 50)) &
                (pl.col("SpeedAvg") > 0.1) & (pl.col("Distance") > 50) & (pl.col("Distance") <= 3600))
          .then(pl.min_horizontal([(pl.col("Distance") / (pl.col("SpeedAvg") / 3.6)), pl.lit(float(900))]).clip(lower_bound=180))
          .otherwise((pl.col("TimeStampFin") - pl.col("TimeStampIni")).dt.total_seconds().abs()).alias("CycleDurationSeconds"),
        pl.when(pl.col("StageSequence") == 1).then(0).otherwise(pl.col("TimeEfficiencyPercentage")).alias("TimeEfficiencyPercentage"),
        pl.when(pl.col("Destination").str.strip_chars().str.len_bytes() > 2).then(pl.col("Destination")).otherwise(pl.lit("UNKNOWN")).alias("Destination"),
        pl.col("DestinationType").fill_null(pl.lit("LoadingZone")).alias("DestinationType"),
        pl.col("Material").fill_null(pl.lit("Empty")).alias("Material"),
        pl.col("Shovel").fill_null(pl.lit("Unknown")).alias("Shovel"),
    ])
    
    result = result.filter((pl.col("StageSequence").is_in([4, 8])) & (pl.col("StageSequence").is_not_null()))
    
    print(f"✓ {len(result):,} ciclos procesados")
    return result

def predict(df, m4, m8):
    print("Generando predicciones...")
    df_pd = df.to_pandas()
    preds = np.zeros(len(df_pd))
    
    mask_4 = df_pd["StageSequence"] == 4
    if mask_4.any():
        X = df_pd.loc[mask_4, FEATURES_STAGE4["numeric"] + FEATURES_STAGE4["categorical"]].copy()
        for c in FEATURES_STAGE4["categorical"]:
            if c in X.columns:
                X[c] = X[c].astype("category")
                if len(X[c].cat.categories) == 0: X[c] = pd.Categorical(['Unknown'] * len(X))
        try:
            preds[mask_4] = m4.predict(X)
            print(f"✓ Stage 4: {mask_4.sum()} predicciones")
        except Exception as e:
            print(f"✗ S4: {e}")
            preds[mask_4] = 0
    
    mask_8 = df_pd["StageSequence"] == 8
    if mask_8.any():
        X = df_pd.loc[mask_8, FEATURES_STAGE8["numeric"] + FEATURES_STAGE8["categorical"]].copy()
        for c in FEATURES_STAGE8["categorical"]:
            if c in X.columns:
                valid = X[c].notna() & (X[c] != '')
                if not valid.all(): X.loc[~valid, c] = 'Unknown'
                X[c] = X[c].astype("category")
                if len(X[c].cat.categories) == 0: X[c] = pd.Categorical(['Unknown'] * len(X))
        try:
            preds[mask_8] = m8.predict(X)
            print(f"✓ Stage 8: {mask_8.sum()} predicciones")
        except Exception as e:
            print(f"✗ S8: {e}")
            preds[mask_8] = 0
    
    return df.with_columns([pl.Series("PredictedFuel", preds)])

def get_metrics():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        print("Obteniendo métricas...")
        exp = mlflow.get_experiment_by_name(f"{TRUCK_ID}_fuel_prediction")
        if not exp: return None
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs.empty: return None
        r = runs.iloc[0]
        metrics = {
            'stage4': {'metrics': {'R2': r.get('metrics.stage4_R2', 0), 'MAE': r.get('metrics.stage4_MAE', 0), 'RMSE': r.get('metrics.stage4_RMSE', 0), 'MAPE_Safe': r.get('metrics.stage4_MAPE', 0), 'MedianAE': 0, 'RMSLE': r.get('metrics.stage4_RMSLE', 0), 'ExplainedVar': 0}, 'samples': {'train': int(r.get('params.train_samples_stage4', 0)), 'test': int(r.get('params.test_samples_stage4', 0))}},
            'stage8': {'metrics': {'R2': r.get('metrics.stage8_R2', 0), 'MAE': r.get('metrics.stage8_MAE', 0), 'RMSE': r.get('metrics.stage8_RMSE', 0), 'MAPE_Safe': r.get('metrics.stage8_MAPE', 0), 'MedianAE': 0, 'RMSLE': r.get('metrics.stage8_RMSLE', 0), 'ExplainedVar': 0}, 'samples': {'train': int(r.get('params.train_samples_stage8', 0)), 'test': int(r.get('params.test_samples_stage8', 0))}}
        }
        print(f"✓ Métricas: S4 R²={metrics['stage4']['metrics']['R2']:.4f}, S8 R²={metrics['stage8']['metrics']['R2']:.4f}")
        return metrics
    except Exception as e:
        print(f"✗ Error métricas: {e}")
        return None


def main():
    output = f"Fuel_Predictions_{TRUCK_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    print("="*60)
    print(f"PREDICCIÓN COMBUSTIBLE - {TRUCK_ID}")
    print(f"Output: {output}")
    print("="*60)
    
    m4, m8 = load_models()
    if not m4 or not m8: return 1

    metrics = get_metrics()
    if not metrics: print("⚠️ Sin métricas")
    
    df = load_data()
    if df is None or df.is_empty(): return 1
    
    df_cycles = transform_cycles(df)
    if df_cycles is None or df_cycles.is_empty(): return 1
    
    df_pred = predict(df_cycles, m4, m8)
    
    print("Guardando CSV...")
    try:
        pdf = ReportGenerator(df=df_pred, truck_id=TRUCK_ID, model_metrics=metrics).generate_pdf_report(output_path=output)
        print(f"✓ Reporte: {output} ({len(pdf):,} bytes)")
        print("="*60)
        print("✓ COMPLETADO")
        print("="*60)
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())