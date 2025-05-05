# transform/implementation/sensor/schema.py
from datetime import datetime, date
from pydantic import BaseModel, Field

class SensorSchema(BaseModel):
    """Esquema Pydantic v2 para validación de datos de sensores"""
    
    ShiftDate: date # campo obligatorio
    Shift: str = Field(default="NaN")
    TimeStamp: datetime # campo obligatorio
    RecordDuration: float = Field(default=0.0, ge=0)
    Equipment: str # campo obligatorio
    TruckFleet: str = Field(default="NaN")
    FuelLevel: float = Field(default=0.0, ge=0)
    FuelLevelLiters: float = Field(default=0.0, ge=0)
    FuelGauge: str = Field(default="NaN")
    Speed: float = Field(default=0.0, ge=0)
    RPM: int = Field(default=0, ge=0)
    Ralenti: str = Field(default="NaN")
    Latitude: float  # coordenadas locales en mm
    Longitude: float # coordenadas locales en mm
    Elevation: float # coordenadas locales en mm
