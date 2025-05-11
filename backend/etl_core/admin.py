from django.contrib import admin
from .models import FuelSupply, Cycle, Sensor, TimeModel


@admin.register(FuelSupply)
class FuelSupplyAdmin(admin.ModelAdmin):
    list_display = ["Veh", "volumCorregido", "fin_desp"]


@admin.register(Cycle)
class CycleAdmin(admin.ModelAdmin):
    list_display = ["Equipment", "TruckFleet"]


@admin.register(Sensor)
class SensorAdmin(admin.ModelAdmin):
    list_display = ["Equipment", "TruckFleet"]


@admin.register(TimeModel)
class TimeModelAdmin(admin.ModelAdmin):
    list_display = ["Equipment", "TruckFleet"]
