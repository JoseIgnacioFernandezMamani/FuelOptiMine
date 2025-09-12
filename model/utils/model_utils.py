import logging
import pandas as pd
import datetime
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def analyze_multicollinearity(
    self,
    cycles_data,
    predictor_vars,
) -> dict:
    """
    Analyze multicollinearity among predictor variables using:
        - Pearson correlation matrix
        - Variance Inflation Factor (VIF)
    Applied directly on self.cycles_data (unscaled data).
    """
    df = cycles_data[predictor_vars].drop_nulls().drop_nans().to_pandas()

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


def log_results(self, stage: str, results: dict):
    """
    Logs all training results in a structured way.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": stage,
        "predictors": self.predictor_vars,
        "results": results,
    }

    # save as json
    logger.info("Training summary:\n%s", json.dumps(log_entry, indent=4, default=str))
