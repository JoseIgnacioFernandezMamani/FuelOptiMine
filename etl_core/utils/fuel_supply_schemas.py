from pydantic import BaseModel, Field
from datetime import date, datetime


class FuelSupplySchema(BaseModel):
    """Pydantic schema for fuel supply data validation"""

    Origin: str  # required field
    ShiftDate: date  # required field
    TimeStamp: datetime  # required field
    Equipment: str  # required field
    TruckFleet: str = Field(default="NaN")
    FuelLevelLiters: float  # required field
    Shift: str = Field(default="NaN")
    FuelLevel: float = Field(default=0.0)
