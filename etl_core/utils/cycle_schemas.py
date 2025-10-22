from datetime import datetime, date
from pydantic import BaseModel, Field
from typing import Optional


class CycleSchema(BaseModel):
    """Esquema para datos de ciclos mineros (estructura bruta)"""

    ShiftDate: date  # required field # fecha del turno
    Shift: str = Field(default="NaN")  # turno
    Shovel: str = Field(default="NaN")  # pala
    ShovelModel: str = Field(default="NaN")  # modelo de la pala
    Equipment: str  # required field # equipo
    TruckFleet: str = Field(default="NaN")  # flota
    LoadingZone: str = Field(default="NaN")  # zona de carga del material
    Material: str = Field(default="NaN")
    MeasuredTonnage: float = Field(default=0.0)
    ReportedTonnage: float = Field(default=0.0)
    DestinationType: str = Field(default="NaN")
    Destination: str = Field(default="NaN")
    TravelingEmpty: float = Field(default=0.0)
    E_TravelingStart: datetime  # tiempo de inicio del viaje vacio
    E_TravelingEnd: datetime  # tiempo de final del viaje vacio
    WaitingEmpty: float = Field(default=0.0)
    E_WaitingStart: datetime  # tiempo de espera vacio
    E_WaitingEnd: datetime  # tiempo que termina la espera vacio
    SpottingEmpty: float = Field(default=0.0)
    E_SpottingStart: datetime  # tiempo de esperando cuadrando
    E_SpottingEnd: datetime  # tiempo que termina la espera cuadrando
    LoadingMaterial: float = Field(default=0.0)
    E_LoadingStart: datetime  # timepo de carguio inicio
    E_LoadingEnd: datetime  # timepo de fin del carguio fin
    Hauling: float = Field(default=0.0)
    L_HaulingStart: datetime  # tiempo viaje lleno
    L_HaulingEnd: datetime  # tiempo de fin del viaje lleno
    WaitingLoad: float = Field(default=0.0)
    L_WaitingStart: datetime  # tiempo de inicio de la cola
    L_WaitingEnd: datetime  # tiempo de fin de la cola
    SpottingLoad: float = Field(default=0.0)
    L_SpottingStart: datetime  # tiempo de inicio de retrocediendo
    L_SpottingEnd: datetime  # tiempo de fin de retrocediendo
    UnloadingMaterial: float = Field(default=0.0)
    L_UnloadingStart: datetime  # tiempo de inicio de descarga
    L_UnloadingEnd: datetime  # tiempo fin de la descarga
    DistanceEmpty: float = Field(default=0.0)
    DistanceLoaded: float = Field(default=0.0)
    G_Latitude: float = Field(
        default=0.0
    )  # local coordinates in milliarcseconds using the WGS84 system
    G_Longitude: float = Field(default=0.0)  # local coordinates in milliarcseconds
    G_Elevation: float = Field(default=0.0)  # local coordinates in centimeters
    D_Latitude: float = Field(default=0.0)  # local coordinates in milliarcseconds
    D_Longitude: float = Field(default=0.0)  # local coordinates in milliarcseconds
    D_Elevation: float = Field(default=0.0)  # local coordinates in centimeters
    EquivalentDistance: float = Field(default=0.0)
    TotalCycleTime: float = Field(default=0.0)
