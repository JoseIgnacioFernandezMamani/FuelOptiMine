from abc import ABC, abstractmethod
from typing import Optional, List, Type, Dict, Any
from datetime import datetime, date
import polars as pl

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

class BaseTransformer(ABC):
    """Clase base con limpieza común mejorada y métricas detalladas"""
    
    def __init__(self):
        self.metrics = {
            'initial_records': 0,
            'cleaned_records': 0,
            'removed_empty_records': 0,
            'removed_null_records': 0,
            'removed_duplicate_records': 0,
            "invalid_schema_records": 0,
            'clean_data_percentage': 0.0
        }

    @property
    @abstractmethod
    def mandatory_columns(self) -> List[str]:
        """Columnas obligatorias definidas por cada subclase"""
        pass

    @property
    @abstractmethod
    def schema_model(self) -> Type[BaseModel]:
        """Subclases deben proveer su modelo Pydantic"""
        pass
    
    @abstractmethod
    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Método abstracto para la transformación de datos"""
        pass

    def _validate_mandatory_columns(self, df: pl.DataFrame) -> None:
        """Valida columnas obligatorias"""
        missing = [col for col in self.mandatory_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columnas críticas faltantes: {missing}")

    def common_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Limpieza común optimizada con métricas precisas"""
        # 1. Validar columnas obligatorias
        self._validate_mandatory_columns(df)
        
        # 2. Inicializar métricas
        initial_count = len(df)
        self.metrics['initial_records'] = initial_count

         # 3. Convertir todos los datos a string de forma segura
        df = df.cast({col: pl.String for col in df.columns})

        df = df.with_columns(
            pl.all().map_elements(
                lambda x: None if (isinstance(x, str) and (x.lower() == "null" or x=="")) else x,
                return_dtype=pl.String 
            )
        )

        # 3. Eliminar registros con valores nulos en columnas obligatorias
        df_clean = df.drop_nulls(subset=self.mandatory_columns)
        self.metrics['removed_null_records'] = initial_count - len(df_clean)
        
        # 4. Eliminar duplicados temporales (si existe la columna)
        ts_duplicates = 0
        if 'TimeStamp' in df_clean.columns:
            before_ts = len(df_clean)
            df_clean = df_clean.unique(subset=['TimeStamp'], keep='first')
            ts_duplicates = before_ts - len(df_clean)
        
        # 5. Eliminar duplicados completos
        before_full = len(df_clean)
        df_clean = df_clean.unique()
        full_duplicates = before_full - len(df_clean)
        
        # 6. Calcular métricas combinadas
        self.metrics['removed_duplicate_records'] = ts_duplicates + full_duplicates
        self.metrics['cleaned_records'] = len(df_clean)
        
        # 7. Calcular porcentaje de limpieza
        if initial_count > 0:
            self.metrics['clean_data_percentage'] = round(
                (self.metrics['cleaned_records'] / initial_count) * 100, 2
            )
        
        return df_clean
    
    def normalize_and_validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normaliza tipos y valida contra el esquema en un solo paso"""
        valid_rows = []
        
        for row in df.iter_rows(named=True):
            try:
                row_data = self._normalize_row(row)
                self.schema_model.model_validate(row_data)
                valid_rows.append(row_data)
            except ValidationError:
                self.metrics['invalid_schema_records'] += 1
        
        return pl.DataFrame(valid_rows)

    def _normalize_row(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """normalizar"""
        normalized = {}
        
        # Iterar sobre (nombre_campo, FieldInfo)
        for field_name, field in self.schema_model.model_fields.items():
            value = raw_data.get(field_name)
            
            # Manejar campos no presentes en los datos crudos
            if value is None and not field.is_required():
                value = field.get_default()
            
            normalized[field_name] = self._cast_value(field, value, field_name)
        
        return normalized

    def _cast_value(self, field: FieldInfo, value: Any, field_name: str) -> Any:
        """Casteo optimizado para Polars"""
        #print(field, value)
        try:
            # Si el valor es nulo y el campo tiene default
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
                return bool(value)
        except (TypeError, ValueError) as e:
            print(f"⚠️ [ERROR] Campo: {field_name} | Valor: {value} | Error: {str(e)}")  
            return field.default

    def _update_final_metrics(self, final_count: int):
        """Actualiza métricas finales"""
        self.metrics['cleaned_records'] = final_count
        if self.metrics['initial_records'] > 0:
            self.metrics['clean_data_percentage'] = round(
                (final_count / self.metrics['initial_records']) * 100, 2
            )


    def run_transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Pipeline completo optimizado"""
        try:
            if df.is_empty():
                print("DataFrame de entrada vacío")
                return None
                
            df_clean = self.common_clean(df)
            print(f"Datos después de limpieza: {len(df_clean)} registros")
            
            df_normalized = self.normalize_and_validate(df_clean)
            print(f"Datos después de normalización: {len(df_normalized)} registros")
            
            df_transformed = self.transform(df_normalized)
            print(f"Datos después de transformación: {len(df_transformed)} registros")
            
            #self._update_final_metrics(len(df_transformed) if df_transformed else 0)
            return df_transformed
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error crítico en transformación: {str(e)}")
            return None