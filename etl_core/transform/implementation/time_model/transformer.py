from typing import List, Type, Optional
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.time_model_schemas import TimeModelSchema
from pydantic import BaseModel
import polars as pl


class TimeModelTransformer(BaseTransformer):
    """Optimized transformer for time model data using Polars"""

    def __init__(self):
        super().__init__()
        self.metrics.update(
            {
                "categorical_empty_fixed": 0,
                "negative_durations_fixed": 0,
                "invalid_status_combinations": 0,
            }
        )
        self.categorical_columns = [
            "Shift",
            "TruckFleet",
            "Event",
        ]

    @property
    def mandatory_columns(self) -> List[str]:
        return [
            field_name
            for field_name, field in TimeModelSchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        return TimeModelSchema

    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Optimized transformation pipeline"""
        # 1. Validate mandatory columns
        self._validate_mandatory_columns(df)

        # 2. Apply transformations in a single step
        df = df.with_columns(
            self._get_categorical_normalization_exprs()
            + self._get_duration_validation_exprs()
            + self._get_status_normalization_exprs()
        )

        # 3. Count corrections
        self._count_categorical_fixes(df)
        self._count_duration_fixes(df)

        # 4. Apply filters without adding columns
        df = self._apply_filters(df)

        # 5. Update final metrics
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

        # 6. Sort results
        return df.sort("ShiftDate", "TimeStamp")

    def _get_categorical_normalization_exprs(self) -> list[pl.Expr]:
        """Basic normalization for categorical fields"""
        return [
            pl.when(
                pl.col(col).is_null() | (pl.col(col) == "") | (pl.col(col) == "NaN")
            )
            .then(pl.lit("NaN"))
            .otherwise(pl.col(col).str.strip_chars().str.to_uppercase())
            .alias(col)
            for col in self.categorical_columns
        ]

    def _get_duration_validation_exprs(self) -> list[pl.Expr]:
        """Simple validation of durations"""
        return [
            pl.when(pl.col("RecordDuration") < 0)
            .then(0.0)
            .otherwise(pl.col("RecordDuration"))
            .alias("RecordDuration")
        ]

    def _get_status_normalization_exprs(self) -> list[pl.Expr]:
        """Normalization of statuses and categories with encoding correction"""
        status_mapping = {
            r"(?i)Operativo": "OPERATIVO",
            r"(?i)Reserva": "RESERVA",
            r"(?i)Demora": "DEMORA",
            r"(?i)Mantenimiento": "MANTENIMIENTO",
        }

        category_mapping = {
            r"(?i)d_no_programada": "D_NO_PROGRAMADA",
            r"(?i)d_programada": "D_PROGRAMADA",
            r"(?i)efectivo": "EFECTIVO",
            r"(?i)m_no_programado": "M_NO_PROGRAMADO",
            r"(?i)m_programado": "M_PROGRAMADO",
            r"(?i)reparacion": "REPARACION",
            r"(?i)reserva": "RESERVA",
        }

        def normalize_text(expr: pl.Expr) -> pl.Expr:
            # Encoding correction
            corrections = {
                "Ã¡": "a",
                "Ã©": "e",
                "Ã³": "o",
                "Ãº": "u",
                "Ã±": "n",
                "Ã": "a",
                "â€": "",
            }
            for wrong, right in corrections.items():
                expr = expr.str.replace(wrong, right, literal=True)

            # Remove accents
            accents = {
                "á": "a",
                "é": "e",
                "í": "i",
                "ó": "o",
                "ú": "u",
                "ñ": "n",
                "Á": "A",
                "É": "E",
                "Í": "I",
                "Ó": "O",
                "Ú": "U",
                "Ñ": "N",
            }
            for accent, replacement in accents.items():
                expr = expr.str.replace(accent, replacement, literal=True)

            return expr.str.to_uppercase()

        return [
            # Status normalization
            pl.coalesce(
                *[
                    pl.when(
                        normalize_text(pl.col("Status")).str.contains(pattern)
                    ).then(pl.lit(value))
                    for pattern, value in status_mapping.items()
                ],
                normalize_text(pl.col("Status"))
            ).alias("Status"),
            # Category normalization
            pl.coalesce(
                *[
                    pl.when(
                        normalize_text(pl.col("Category")).str.contains(pattern)
                    ).then(pl.lit(value))
                    for pattern, value in category_mapping.items()
                ],
                normalize_text(pl.col("Category"))
            ).alias("Category"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count corrections in categorical fields"""
        all_categorical = self.categorical_columns + ["Status", "Category"]
        for col in all_categorical:
            if col in df.columns:
                null_count = df.filter(
                    pl.col(col).is_in(["", "NaN"]) | pl.col(col).is_null()
                ).height
                self.metrics["categorical_empty_fixed"] += null_count

    def _count_duration_fixes(self, df: pl.DataFrame) -> None:
        """Count corrected negative durations"""
        if "RecordDuration" in df.columns:
            negative_count = df.filter(pl.col("RecordDuration") < 0).height
            self.metrics["negative_durations_fixed"] += negative_count

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter invalid combinations without adding columns"""
        before_filter = df.height

        invalid_combinations = (pl.col("Status") == "OPERATIVO") & (
            pl.col("Category").is_in(["D_NO_PROGRAMADA", "D_PROGRAMADA"])
        ) | (pl.col("Status") == "MANTENIMIENTO") & (
            ~pl.col("Category").str.contains("MANTENIMIENTO")
        )

        df = df.filter(~invalid_combinations)
        self.metrics["invalid_status_combinations"] = before_filter - df.height
        return df
