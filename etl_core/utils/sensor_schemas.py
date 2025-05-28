# transform/implementation/sensor/schema.py
from datetime import datetime, date
from pydantic import BaseModel, Field


class SensorSchema(BaseModel):
    """Esquema Pydantic v2 para validación de datos de sensores
    degree=milliarcseconds/3600000 -> ej latitude -75968688/3600000 = -21.05, longitude -241943798/3600000 = -67.2066105557
    google maps: (-21.102413333, -67.21) -> (lat, lon), Nota: nunca redondear los datos, siempre usar el valor completo del resultado
    """

    ShiftDate: date  # campo obligatorio
    Shift: str = Field(default="NaN")
    TimeStamp: datetime  # campo obligatorio
    RecordDuration: float = Field(default=0.0, ge=0)
    Equipment: str  # campo obligatorio
    TruckFleet: str = Field(default="NaN")
    FuelLevel: float = Field(default=0.0, ge=0)
    FuelLevelLiters: float = Field(default=0.0, ge=0)
    FuelGauge: str = Field(default="NaN")
    Speed: float = Field(default=0.0, ge=0)
    RPM: float = Field(default=0.0, ge=0)
    Ralenti: str = Field(default="NaN")
    Latitude: float  # coordenadas locales milliarcseconds usando el sistema WGS84
    Longitude: float  # coordenadas locales milliarcseconds
    Elevation: float  # coordenadas locales milliarcseconds
