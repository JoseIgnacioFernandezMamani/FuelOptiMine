# utils/validators.py
import pandas as pd
from typing import Dict, Any, List, Optional

class DataFrameValidator:
    """Validador para DataFrames."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Inicializa con un schema que especifica los tipos y restricciones.
        
        schema = {
            'column_name': {
                'type': 'int64',
                'nullable': False,
                'min': 0,
                'max': 100
            }
        }
        """
        self.schema = schema
    
    def validate(self, df: pd.DataFrame) -> List[str]:
        """Valida un DataFrame contra el schema definido."""
        errors = []
        
        # Verificar columnas requeridas
        for column, specs in self.schema.items():
            if column not in df.columns:
                errors.append(f"Falta la columna: {column}")
                continue
            
            # Verificar tipo de datos
            if 'type' in specs and df[column].dtype != specs['type']:
                errors.append(f"Tipo incorrecto en {column}: esperado {specs['type']}, obtenido {df[column].dtype}")
            
            # Verificar valores nulos
            if 'nullable' in specs and not specs['nullable'] and df[column].isnull().any():
                errors.append(f"La columna {column} no debe contener valores nulos")
            
            # Verificar rango
            if 'min' in specs and df[column].min() < specs['min']:
                errors.append(f"Valor menor al mínimo en {column}: {df[column].min()} < {specs['min']}")
            
            if 'max' in specs and df[column].max() > specs['max']:
                errors.append(f"Valor mayor al máximo en {column}: {df[column].max()} > {specs['max']}")
        
        return errors