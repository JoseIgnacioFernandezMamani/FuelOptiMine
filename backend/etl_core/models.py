from django.db import models
from .model_builder import pydantic_to_django_field, create_django_model
from etl_core.etl.transform.implementation.fuel_supply.schema import FuelSupplySchema
from etl_core.etl.transform.implementation.cycle.schema import CycleSchema
from etl_core.etl.transform.implementation.sensor.schema import SensorSchema
from etl_core.etl.transform.implementation.time_model.schema import TimeModelSchema

FuelSupply = create_django_model(FuelSupplySchema, "FuelSupply")
Cycle = create_django_model(CycleSchema, "Cycle")
Sensor = create_django_model(SensorSchema, "Sensor")
TimeModel = create_django_model(TimeModelSchema, "TimeModel")
