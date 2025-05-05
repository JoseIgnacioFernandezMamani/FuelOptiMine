from pydantic import BaseModel, Field
from datetime import datetime

class FuelSupplySchema(BaseModel):
    """Esquema para datos de despacho minero"""
    
    # Todos los campos son obligatorios
    Veh: str
    Descripcion: str
    fin_desp: datetime
    volumCorregido: float 
    Origin: str
