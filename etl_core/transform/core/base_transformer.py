from abc import ABC, abstractmethod
from typing import Optional, List, Type, Dict, Any
from datetime import datetime, date
import polars as pl

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo


class BaseTransformer(ABC):
    """
    Base class for ETL data transformers.
    Actions:
    1. Data cleaning: remove null or inconsistent records
    2. Data type casting: convert to correct data types
    3. Column normalization: convert to correct units
    4. Integrity validation: remove duplicate data
    5. Enrichment: add metadata and metrics
    """

    def __init__(self):

        # 5. Enrichment: add metadata and metrics
        self.metrics = {
            "initial_records": 0,
            "after_cleaning_records": 0,
            "after_validation_records": 0,
            "after_transform_records": 0,
            "removed_empty_records": 0,
            "removed_null_records": 0,
            "removed_duplicate_records": 0,
            "invalid_schema_records": 0,
            "clean_data_percentage": 0.0,
            "valid_data_percentage": 0.0,
            "final_data_percentage": 0.0,
        }

    """Abstract methods and common properties that must be implemented by subclasses"""

    @property
    @abstractmethod
    def mandatory_columns(self) -> List[str]:
        """Mandatory columns defined by each subclass"""
        pass

    @property
    @abstractmethod
    def schema_model(self) -> Type[BaseModel]:
        """Subclasses must provide their Pydantic model"""
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Abstract method for data transformation"""
        pass

    # 4. integrity validation
    def _validate_mandatory_columns(self, df: pl.DataFrame) -> None:
        """Validate mandatory columns"""
        missing = [col for col in self.mandatory_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columnas críticas faltantes: {missing}")

    # 1. data cleaning, remove null or inconsistent records and
    def common_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimized common cleaning with accurate metrics"""
        # 1. Validate mandatory columns
        self._validate_mandatory_columns(df)

        # 2. Initialize metrics
        initial_count = len(df)
        self.metrics["initial_records"] = initial_count

        # 3. Safely convert all data to string
        df = df.cast({col: pl.String for col in df.columns})

        # count empty records before cleaning
        empty_count = df.filter(
            pl.any_horizontal(pl.col("*").is_null() | (pl.col("*") == ""))
        ).height
        self.metrics["removed_empty_records"] = empty_count

        # replace null or "null" values in all columns
        df = df.with_columns(
            pl.all().map_elements(
                lambda x: (
                    None
                    if (isinstance(x, str) and (x.lower() == "null" or x == ""))
                    else x
                ),
                return_dtype=pl.String,
            )
        )

        # 4. Remove records with null values in mandatory columns
        df_clean = df.drop_nulls(subset=self.mandatory_columns)
        self.metrics["removed_null_records"] = initial_count - len(df_clean)

        # 5. Remove temporary duplicates (if the column exists)
        ts_duplicates = 0
        if "TimeStamp" in df_clean.columns:
            before_ts = len(df_clean)
            df_clean = df_clean.unique(subset=["TimeStamp"], keep="first")
            ts_duplicates = before_ts - len(df_clean)

        # 6. Remove full duplicates
        before_full = len(df_clean)
        df_clean = df_clean.unique()
        full_duplicates = before_full - len(df_clean)

        # 7. Update metrics
        self.metrics["removed_duplicate_records"] = ts_duplicates + full_duplicates
        self.metrics["after_cleaning_records"] = len(df_clean)

        # 8. Calcular porcentaje de limpieza
        if initial_count > 0:
            self.metrics["clean_data_percentage"] = round(
                (self.metrics["after_cleaning_records"] / initial_count) * 100, 2
            )

        return df_clean

    def normalize_and_validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize types and validate against the schema in a single step"""
        valid_rows = []
        invalid_count = 0

        for row in df.iter_rows(named=True):
            try:
                row_data = self._normalize_row(row)
                self.schema_model.model_validate(row_data)
                valid_rows.append(row_data)
            except ValidationError:
                invalid_count += 1

        # Update metrics
        self.metrics["invalid_schema_records"] = invalid_count
        df_valid = pl.DataFrame(valid_rows)
        self.metrics["after_validation_records"] = len(df_valid)

        # Calculate valid data percentage
        if self.metrics["after_cleaning_records"] > 0:
            self.metrics["valid_data_percentage"] = round(
                (
                    self.metrics["after_validation_records"]
                    / self.metrics["after_cleaning_records"]
                )
                * 100,
                2,
            )

        return df_valid

    # 3. column normalization, convert to correct units
    def _normalize_row(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """normalize"""
        normalized = {}

        # Iterar sobre (nombre_campo, FieldInfo)
        for field_name, field in self.schema_model.model_fields.items():
            value = raw_data.get(field_name)

            # Manejar campos no presentes en los datos crudos
            if value is None and not field.is_required():
                value = field.get_default()

            normalized[field_name] = self._cast_value(field, value, field_name)

        return normalized

    # 2. data type casting, convert to correct data types and
    def _cast_value(self, field: FieldInfo, value: Any, field_name: str) -> Any:
        """Casting for Polars"""
        try:
            # If the value is null and the field has a default
            if value is None and not field.is_required():
                return field.get_default()
            if field.annotation == date:
                return datetime.strptime(value, "%Y-%m-%d").date()
            elif field.annotation == datetime:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            elif field.annotation == float:
                return float(value)
            elif field.annotation == int:
                return int(value)
            elif field.annotation == str:
                return str(value)
            elif field.annotation == bool:
                if isinstance(value, str):
                    value = value.strip().lower()
                    if value in ["true", "1", "yes", "y", "t"]:
                        return True
                    elif value in ["false", "0", "no", "n", "f"]:
                        return False
                    else:
                        raise ValueError(f"Valor booleano no válido: {value}")
                else:
                    return bool(value)
        except (TypeError, ValueError) as e:
            print(f"⚠️ [ERROR] Campo: {field_name} | Valor: {value} | Error: {str(e)}")
            return field.default

    def run_transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Pipeline completo optimizado"""
        try:
            if df.is_empty():
                print("Empty input DataFrame")
                return None

            # initialize cleaning metrics
            df_clean = self.common_clean(df)
            print(
                f"Datos después de limpieza: {self.metrics['after_cleaning_records']} registros"
            )

            # validate
            df_normalized = self.normalize_and_validate(df_clean)
            print(
                f"registros validos: {self.metrics["after_validation_records"]} registros"
            )
            print(
                f"registros invalidos: {self.metrics['invalid_schema_records']} registros"
            )

            # transform
            df_transformed = self.transform(df_normalized)
            final_count = len(df_transformed) if df_transformed.is_empty() else 0
            self.metrics["after_transform_records"] = final_count

            # calculate final data percentage
            if self.metrics["initial_records"] > 0:
                self.metrics["final_data_percentage"] = round(
                    (
                        self.metrics["after_transform_records"]
                        / self.metrics["initial_records"]
                    )
                    * 100,
                    2,
                )

            print(
                f"Registros finales: {self.metrics['after_transform_records']} registros"
            )
            print(
                f"Porcentaje de datos finales: {self.metrics['final_data_percentage']}%"
            )

            return df_transformed

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error during transformation: {str(e)}")
            print(f"Current metrics: {self.metrics}")
            print(
                f"Dataframe schema: {df.schema if df is not None else 'No dataframe'}"
            )
            return None
