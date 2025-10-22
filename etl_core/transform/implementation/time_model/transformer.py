from typing import List, Type, Optional
from etl_core.transform.core import BaseTransformer
from etl_core.utils.time_model_schemas import TimeModelSchema
from pydantic import BaseModel
import polars as pl
from polars import Expr


class TimeModelTransformer(BaseTransformer):
    """Optimized transformer for time model data using Polars"""

    def __init__(self):
        super().__init__()
        self.metrics.update(
            {
                "categorical_empty_fixed": 0,
                "negative_durations_fixed": 0,
            }
        )
        self.categorical_columns = ["Shift", "TruckFleet"]

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

        # 5. Update final metrics
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

        # 6. Sort results
        return df.sort("ShiftDate", "TimeStamp").with_row_index("TimeModelId", offset=0)

    def normalize_text(self, expr: pl.Expr) -> pl.Expr:
        """Centralized text normalization function"""
        # Encoding correction
        corrections: dict[str, str] = {
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
        accents: dict[str, str] = {
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
        status_mapping: dict[str, str] = {
            r"(?i)Operativo": "OPERATIVO",
            r"(?i)Reserva": "RESERVA",
            r"(?i)Demora": "DEMORA",
            r"(?i)Mantenimiento": "MANTENIMIENTO",
        }

        category_mapping: dict[str, str] = {
            r"(?i)d_no_programada": "D_NO_PROGRAMADA",
            r"(?i)d_programada": "D_PROGRAMADA",
            r"(?i)efectivo": "EFECTIVO",
            r"(?i)m_no_programado": "M_NO_PROGRAMADO",
            r"(?i)m_programado": "M_PROGRAMADO",
            r"(?i)reparacion": "REPARACION",
            r"(?i)reserva": "RESERVA",
        }

        event_mapping: dict[str, str] = {
            r"(?i)inspección": "INSPECCION",
            r"(?i)voladura": "VOLADURA",
            r"(?i)reunión": "REUNION",
            r"(?i)capacitación": "CAPACITACION",
            r"(?i)condiciones climáticas": "CLIMA",
            r"(?i)después de mtto": "POST_MANTENIMIENTO",
            r"(?i)disip.*gases": "DISIPACION_GASES",
            r"(?i)obstrucción de vía": "OBSTRUCCION_VIA",
            r"(?i)topografía": "TOPOGRAFIA",
            r"(?i)challa": "CHALLA",
            r"(?i)empantanado": "EMPANTANADO",
            r"(?i)rev[ií]sión médica": "REVISION_MEDICA",
            r"(?i)almuerzo/cena": "ALMUERZO",
            r"(?i)punto entel \(wc\)": "WC_ENTEL",
            r"(?i)esperando.*pala": "ESPERANDO_PALA",
            r"(?i)esperando.*cam[ií]ón": "ESPERANDO_CAMION",
            r"(?i)falla hxgn.*oas": "FALLA_OAS",
            r"(?i)falla hxgn.*fms": "FALLA_FMS",
            r"(?i)falla hxgn.*cas10": "FALLA_CAS10",
            r"(?i)falla.*i-track": "FALLA_ITRACK",
            r"(?i)traslado": "TRASLADO",
            r"(?i)relleno.*combustible": "RELLENO_COMBUSTIBLE",
            r"(?i)cambio.*operador": "CAMBIO_OPERADOR",
            r"(?i)cambio.*turno": "CAMBIO_TURNO",
            r"(?i)sin asignación": "SIN_ASIGNACION",
            r"(?i)restricci[oó]n.*voladura": "RESTRICCION_VOLADURA",
            r"(?i)taponamiento": "TAPONAMIENTO",
            r"(?i)c[oó]digo.*eliminado|999": "ELIMINADO",
        }

        return [
            # Status normalization
            pl.coalesce(
                *[
                    pl.when(
                        self.normalize_text(pl.col("Status")).str.contains(pattern)
                    ).then(pl.lit(value))
                    for pattern, value in status_mapping.items()
                ],
                self.normalize_text(pl.col("Status"))
            ).alias("Status"),
            # Category normalization
            pl.coalesce(
                *[
                    pl.when(
                        self.normalize_text(pl.col("Category")).str.contains(pattern)
                    ).then(pl.lit(value))
                    for pattern, value in category_mapping.items()
                ],
                self.normalize_text(pl.col("Category"))
            ).alias("Category"),
            # Event normalization
            pl.coalesce(
                *[
                    pl.when(
                        self.normalize_text(pl.col("Event")).str.contains(pattern)
                    ).then(
                        self.normalize_text(pl.col("Event")).str.replace(pattern, value)
                    )
                    for pattern, value in event_mapping.items()
                ],
                self.normalize_text(pl.col("Event"))
            ).alias("Event"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count corrections in categorical fields"""
        all_categorical: list[str] = self.categorical_columns + [
            "Status",
            "Category",
            "Event",
        ]
        for col in all_categorical:
            if col in df.columns:
                null_count: int = df.filter(
                    pl.col(col).is_in(["", "NaN"]) | pl.col(col).is_null()
                ).height
                self.metrics["categorical_empty_fixed"] += null_count

    def _count_duration_fixes(self, df: pl.DataFrame) -> None:
        """Count corrected negative durations"""
        if "RecordDuration" in df.columns:
            negative_count: int = df.filter(pl.col("RecordDuration") < 0).height
            self.metrics["negative_durations_fixed"] += negative_count
