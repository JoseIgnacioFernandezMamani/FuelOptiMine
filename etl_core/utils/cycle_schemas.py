from datetime import datetime, date
from pydantic import BaseModel, Field
from typing import Optional


class CycleSchema(BaseModel):
    """Esquema para datos de ciclos mineros (estructura bruta)"""

    ShiftDate: date  # required field
    Shift: str = Field(default="NaN")
    Shovel: str = Field(default="NaN")
    ShovelModel: str = Field(default="NaN")
    Equipment: str  # required field
    TruckFleet: str = Field(default="NaN")
    LoadingZone: str = Field(default="NaN")
    Material: str = Field(default="NaN")
    MeasuredTonnage: float = Field(default=0.0)
    ReportedTonnage: float = Field(default=0.0)
    DestinationType: str = Field(default="NaN")
    Destination: str = Field(default="NaN")
    TravelingEmpty: float = Field(default=0.0)
    E_TravelingStart: datetime = None
    E_TravelingEnd: datetime = None
    WaitingEmpty: float = Field(default=0.0)
    E_WaitingStart: datetime = None
    E_WaitingEnd: datetime = None
    SpottingEmpty: float = Field(default=0.0)
    E_SpottingStart: datetime = None
    E_SpottingEnd: datetime = None
    LoadingMaterial: float = Field(default=0.0)
    E_LoadingStart: datetime = None
    E_LoadingEnd: datetime = None
    Hauling: float = Field(default=0.0)
    L_HaulingStart: datetime = None
    L_HaulingEnd: datetime = None
    WaitingLoad: float = Field(default=0.0)
    L_WaitingStart: datetime = None
    L_WaitingEnd: datetime = None
    SpottingLoad: float = Field(default=0.0)
    L_SpottingStart: datetime = None
    L_SpottingEnd: datetime = None
    UnloadingMaterial: float = Field(default=0.0)
    L_UnloadingStart: datetime = None
    L_UnloadingEnd: datetime = None
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
