from datetime import datetime, date
from pydantic import BaseModel, Field


class TimeModelSchema(BaseModel):
    """Esquema para eventos de operación minera"""

    ShiftDate: date  # Campos obligatorios
    Shift: str = Field(default="NaN")
    TimeStamp: datetime  # Campos obligatorios
    RecordDuration: float = Field(default=0.0)
    Equipment: str  # Campos obligatorios
    TruckFleet: str = Field(default="NaN")
    Status: str = Field(default="NaN")
    Category: str = Field(default="NaN")
    Event: str = Field(default="NaN")
