from django.db import models
from pydantic import BaseModel
from datetime import datetime
from typing import Any, Type, Dict


def pydantic_to_django_field(field_type: Type) -> models.Field:
    """Mapea tipos de Pydantic a campos de Django."""
    type_mapping = {
        str: models.CharField(max_length=255),
        int: models.IntegerField(),
        float: models.FloatField(),
        datetime: models.DateTimeField(),
        bool: models.BooleanField(),
        # Añade más tipos según necesites
    }
    return type_mapping.get(field_type, models.TextField())


def create_django_model(schema: Type[BaseModel], model_name: str) -> type:
    """Crea un modelo Django a partir de un esquema Pydantic."""
    fields = {}

    for field_name, field in schema.model_fields.items():
        django_field = pydantic_to_django_field(field.annotation)

        # Añadir parámetros adicionales (ej: max_length)
        if hasattr(field, "max_length"):
            django_field.max_length = field.max_length

        fields[field_name] = django_field

    # Crear la clase del modelo
    return type(
        model_name,
        (models.Model,),
        {"__module__": __name__, **fields},
    )
