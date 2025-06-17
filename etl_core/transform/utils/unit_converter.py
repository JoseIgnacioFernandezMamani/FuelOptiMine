import polars as pl


def get_coordinate_conversion_exprs(
    lat_col: str = "Latitude", lon_col: str = "Longitude", elev_col: str = "Elevation"
) -> list[pl.Expr]:
    """Devuelve expresiones para conversión de coordenadas"""
    return [
        (pl.col(lat_col) / 3600000).alias(lat_col),
        (pl.col(lon_col) / 3600000).alias(lon_col),
        (pl.col(elev_col) / 100).alias(elev_col),
    ]


def get_geo_validation_expr(
    lat_col: str = "Latitude", lon_col: str = "Longitude", elev_col: str = "Elevation"
) -> pl.Expr:
    """Devuelve expresión para validar rangos geográficos"""
    return (
        pl.col(lat_col).is_between(-90, 90)
        & pl.col(lon_col).is_between(-180, 180)
        & pl.col(elev_col).is_between(0, 8000)
    )
