from datetime import datetime, date
from pydantic import BaseModel, Field


class SensorSchema(BaseModel):
    """Esquema Pydantic v2 para validación de datos de sensores
    degree=milliarcseconds/3600000 -> ej latitude -75968688/3600000 = -21.05, longitude -241943798/3600000 = -67.2066105557
    google maps: (-21.102413333, -67.21) -> (lat, lon), Nota: nunca redondear los datos, siempre usar el valor completo del resultado
    los metros estan en centimetros, por lo que se debe dividir por 100 para obtener metros.
    """

    ShiftDate: date  # required field
    Shift: str = Field(default="NaN")
    TimeStamp: datetime  # required field
    RecordDuration: float = Field(default=0.0)
    Equipment: str  # required field
    TruckFleet: str = Field(default="NaN")
    FuelLevel: float = Field(default=0.0)
    FuelLevelLiters: float = Field(default=0.0)
    FuelGauge: str = Field(default="NaN")
    Speed: float = Field(default=0.0)
    RPM: float = Field(default=0.0)
    Ralenti: str = Field(default="NaN")
    Latitude: float  # local coordinates in milliarcseconds using the WGS84 system, required field
    Longitude: float  # local coordinates in milliarcseconds, required field
    Elevation: float  # local coordinates in centimeters, required field
