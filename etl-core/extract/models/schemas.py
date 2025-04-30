COLUMN_MAPPING = {
    "sensor": [
        'ShiftDate', 'Shift', 'TimeStamp', 'RecordDuration', 'Equipment',
        'TruckFleet', 'FuelLevel', 'FuelLevelLiters', 'FuelGauge', 'Speed',
        'RPM', 'Ralenti', 'Latitude', 'Longitude', 'Elevation'
    ],
    "time_model": [
        'ShiftDate', 'Shift', 'TimeStamp', 'RecordDuration', 'Equipment',
        'TruckFleet', 'Status', 'Category', 'Event'
    ],
    "cycle": [ 
        "ShiftDate", "Shift", "Shovel", "ShovelModel", "Equipment", "TruckFleet", "LoadingZone", "Material", "MeasuredTonnage",
        "ReportedTonnage", "DestinationType", "Destination", "TravelingEmpty", "E_TravelingStart", "E_TravelingEnd", "WaitingEmpty", "E_WaitingStart",
        "E_WaitingEnd", "SpottingEmpty", "E_SpottingStart", "E_SpottingEnd", "LoadingMaterial", "E_LoadingStart", "E_LoadingEnd",
        "Hauling", "L_HaulingStart", "L_HaulingEnd", "WaitingLoad", "L_WaitingStart", "L_WaitingEnd", "SpottingLoad", "L_SpottingStart", "L_SpottingEnd",
        "UnloadingMaterial", "L_UnloadingStart", "L_UnloadingEnd", "DistanceEmpty", "DistanceLoaded", "G_Latitude", "G_Longitude", "G_Elevation", "D_Latitude",
        "D_Longitude", "D_Elevation"
    ]
}

SUPPORTED_FORMATS = {
    "tabular": [".csv", ".tsv", ".parquet", ".feather", ".xls", ".xlsx"],
    "hierarchical": [".json", ".yaml", ".xml"],
    "binary_columnar": [".parquet", ".feather", ".orc"]
}

DATASET_TYPES = [ "test_data", "val_data", "train_data" ]
