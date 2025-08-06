CREATE_TABLE_LSTM_FUEL = """
CREATE TABLE IF NOT EXISTS fuel_optimine.lstm_fuel
(
    -- key fields
    Equipment LowCardinality(String) CODEC(ZSTD(1)),
    TimeStamp DateTime64(3, 'UTC') CODEC(Delta(8), ZSTD(1)),
    ShiftDate Date CODEC(Delta(2), ZSTD(1)),
    
    -- common fields
    Shift LowCardinality(String) CODEC(ZSTD(1)),
    TruckFleet LowCardinality(String) CODEC(ZSTD(1)),
    
    -- sensor fields
    RecordDuration Float32 CODEC(Gorilla, ZSTD(1)),
    FuelLevel Float32 CODEC(Gorilla, ZSTD(1)),
    FuelLevelLiters Float32 CODEC(Gorilla, ZSTD(1)),
    FuelGauge LowCardinality(String) CODEC(ZSTD(1)),
    Speed Float32 CODEC(Gorilla, ZSTD(1)),
    RPM Float32 CODEC(Gorilla, ZSTD(1)),
    Ralenti LowCardinality(String) CODEC(ZSTD(1)),
    Latitude Float64 CODEC(Gorilla, ZSTD(1)),
    Longitude Float64 CODEC(Gorilla, ZSTD(1)),
    Elevation Float32 CODEC(Gorilla, ZSTD(1)),
    SlopePercent Float32 CODEC(Gorilla, ZSTD(1)),
    DistanceTraveled Float32 CODEC(Gorilla, ZSTD(1)),
    
    -- agregation fields 
    Origin LowCardinality(String) CODEC(ZSTD(1)),
    RefillEvent UInt8 CODEC(RLE),
    RefillFuelLevelLiters Float32 CODEC(Gorilla, ZSTD(1)),
    AfterAvgFuelLevelLiters Float32 CODEC(Gorilla, ZSTD(1)),
    BeforeAvgFuelLevelLiters Float32 CODEC(Gorilla, ZSTD(1)),
    TimeDiscrepancy Float32 CODEC(Gorilla, ZSTD(1)),
    RefillDiscrepancy Float32 CODEC(Gorilla, ZSTD(1)),

    -- Time model fields
    Status LowCardinality(String) CODEC(ZSTD(1)),
    Category LowCardinality(String) CODEC(ZSTD(1)),
    Event LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Cycle fields
    Shovel LowCardinality(String) CODEC(ZSTD(1)),
    ShovelModel LowCardinality(String) CODEC(ZSTD(1)),
    LoadingZone LowCardinality(String) CODEC(ZSTD(1)),
    Material LowCardinality(String) CODEC(ZSTD(1)),
    MeasuredTonnage Float32 CODEC(Gorilla, ZSTD(1)),
    ReportedTonnage Float32 CODEC(Gorilla, ZSTD(1)),
    DestinationType LowCardinality(String) CODEC(ZSTD(1)),
    Destination LowCardinality(String) CODEC(ZSTD(1)),
    TravelingEmpty Float32 CODEC(Gorilla, ZSTD(1)),
    WaitingEmpty Float32 CODEC(Gorilla, ZSTD(1)),
    SpottingEmpty Float32 CODEC(Gorilla, ZSTD(1)),
    LoadingMaterial Float32 CODEC(Gorilla, ZSTD(1)),
    Hauling Float32 CODEC(Gorilla, ZSTD(1)),
    WaitingLoad Float32 CODEC(Gorilla, ZSTD(1)),
    SpottingLoad Float32 CODEC(Gorilla, ZSTD(1)),
    UnloadingMaterial Float32 CODEC(Gorilla, ZSTD(1)),
    DistanceEmpty Float32 CODEC(Gorilla, ZSTD(1)),
    DistanceLoaded Float32 CODEC(Gorilla, ZSTD(1)),
    G_Latitude Float64 CODEC(Gorilla, ZSTD(1)),
    G_Longitude Float64 CODEC(Gorilla, ZSTD(1)),
    G_Elevation Float32 CODEC(Gorilla, ZSTD(1)),
    D_Latitude Float64 CODEC(Gorilla, ZSTD(1)),
    D_Longitude Float64 CODEC(Gorilla, ZSTD(1)),
    D_Elevation Float32 CODEC(Gorilla, ZSTD(1)),
    EquivalentDistance Float32 CODEC(Gorilla, ZSTD(1)),
    TotalCycleTime Float32 CODEC(Gorilla, ZSTD(1)),
    
    -- Índices avanzados
    INDEX idx_equipment Equipment TYPE set(0) GRANULARITY 4,
    INDEX idx_timestamp TimeStamp TYPE minmax GRANULARITY 4,
    INDEX idx_status Status TYPE bloom_filter GRANULARITY 2
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(TimeStamp)
ORDER BY (Equipment, TimeStamp)
PRIMARY KEY (Equipment, TimeStamp)
SETTINGS 
    index_granularity = 8192,
    storage_policy = 'hot_cold',
    allow_nullable_key = 1
"""
