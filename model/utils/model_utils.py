import logging
import pandas as pd
import polars as pl
import datetime
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys


def analyze_multicollinearity(
    cycles_data: pl.DataFrame,
    predictor_vars: list[str],
) -> dict:
    """
    Analyze multicollinearity among predictor variables using:
        - Pearson correlation matrix
        - Variance Inflation Factor (VIF)
    Applied directly on self.cycles_data (unscaled data).
    """
    aux = predictor_vars.copy()
    # exception xgboost model
    if "StageSequence" in aux:
        aux.remove("StageSequence")

    df = cycles_data[aux].drop_nulls().drop_nans().to_pandas()

    # Correlation matrix
    corr_matrix = df.corr(method="pearson").to_dict()

    # VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data = {
        df.columns[i]: variance_inflation_factor(df.values, i)
        for i in range(df.shape[1])
    }

    return {"correlation_matrix": corr_matrix, "vif": vif_data}


def log_results(predictor_vars: list[str], stage: str, results: dict, logger):
    """
    Logs all training results in a structured way.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": stage,
        "predictors": predictor_vars,
        "results": results,
    }

    # save as json
    logger.info("Training summary:\n%s", json.dumps(log_entry, indent=4, default=str))


def get_logger(name: str, log_file: str, console: bool = True):
    """
    Crear logger específico por clase con configuración consistente

    Args:
        name: Nombre del logger (ej: "LinearRegression", "XGBoost")
        log_file: Archivo de log (ej: "lrm.log", "xgb.log")
        console: Si mostrar en consola o no
    """
    logger = logging.getLogger(name)

    # Solo configurar si no tiene handlers (evita duplicados)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Handler para archivo
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Handler para consola (opcional)
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(console_handler)

        # Evitar propagación al root logger
        logger.propagate = False

    return logger
