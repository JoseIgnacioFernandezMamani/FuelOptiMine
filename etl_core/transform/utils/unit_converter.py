import polars as pl


def get_coordinate_conversion_exprs() -> list[pl.Expr]:
    """Devuelve expresiones para conversión de coordenadas"""
    return [
        (pl.col("Latitude") / 3600000).alias("Latitude"),
        (pl.col("Longitude") / 3600000).alias("Longitude"),
        (pl.col("Elevation") / 1000).alias("Elevation"),
    ]


def get_geo_validation_expr() -> pl.Expr:
    """Devuelve expresión para validar rangos geográficos"""
    return pl.col("Latitude").is_between(-90, 90) & pl.col("Longitude").is_between(
        -180, 180
    )
