import polars as pl
from typing import List


def get_categorical_normalization_exprs(
    columns: List[str], default_value: str = "NoData"
) -> List[pl.Expr]:
    """
    Generate expressions to normalize categorical columns by replacing null/empty values

    Args:
        columns: List of column names to normalize
        default_value: Value to use for null/empty values

    Returns:
        List of Polars expressions for normalization
    """
    return [
        pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
        .then(pl.lit(default_value))
        .otherwise(pl.col(col))
        .alias(col)
        for col in columns
    ]


def count_null_empty_categorical_values(
    df: pl.DataFrame, columns: List[str], default_value: str = "NoData"
) -> int:
    """
    Count how many categorical values were fixed (null/empty replaced)

    Args:
        df: Transformed DataFrame to analyze
        columns: List of categorical column names
        default_value: The value used to replace null/empty values

    Returns:
        Total count of fixed values
    """
    total_fixed = 0
    for col in columns:
        if col in df.columns:
            # Simply count records that now have the default value
            total_fixed += df.filter(pl.col(col) == default_value).height
    return total_fixed
