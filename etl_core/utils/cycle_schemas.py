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
    E_TravelingStart: Optional[datetime] = None
    E_TravelingEnd: Optional[datetime] = None
    WaitingEmpty: float = Field(default=0.0)
    E_WaitingStart: Optional[datetime] = None
    E_WaitingEnd: Optional[datetime] = None
    SpottingEmpty: float = Field(default=0.0)
    E_SpottingStart: Optional[datetime] = None
    E_SpottingEnd: Optional[datetime] = None
    LoadingMaterial: float = Field(default=0.0)
    E_LoadingStart: Optional[datetime] = None
    E_LoadingEnd: Optional[datetime] = None
    Hauling: float = Field(default=0.0)
    L_HaulingStart: Optional[datetime] = None
    L_HaulingEnd: Optional[datetime] = None
    WaitingLoad: float = Field(default=0.0)
    L_WaitingStart: Optional[datetime] = None
    L_WaitingEnd: Optional[datetime] = None
    SpottingLoad: float = Field(default=0.0)
    L_SpottingStart: Optional[datetime] = None
    L_SpottingEnd: Optional[datetime] = None
    UnloadingMaterial: float = Field(default=0.0)
    L_UnloadingStart: Optional[datetime] = None
    L_UnloadingEnd: Optional[datetime] = None
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
