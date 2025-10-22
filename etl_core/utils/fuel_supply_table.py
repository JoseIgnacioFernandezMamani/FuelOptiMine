CREATE_TABLE_FUEL_SUPPLY = """
-- =================================================================
-- Esta tabla almacena registros procesados de eventos de de recarga de combustible por parte de surtidor y camiones SST/P68
-- =================================================================

CREATE TABLE IF NOT EXISTS fuel_optimine.fuel_supply
(
    -- key field
    FuelSupplyId UInt64
        COMMENT 'Identificador secuencial único del evento de suministro de combustible, generado en el pipeline ETL.'
        CODEC(Delta, ZSTD(1)),

    -- event date fields
    ShiftDate Date
        COMMENT 'Fecha del turno de trabajo (ej: 2025-01-01).'
        CODEC(Delta(2), ZSTD(1)),

    TimeStamp DateTime64(3, 'UTC')
        COMMENT 'Marca temporal precisa del evento (UTC). Punto de referencia temporal para análisis de series de tiempo.'
        CODEC(Delta(8), ZSTD(1)),

    -- common fields
    Shift LowCardinality(String)
        COMMENT 'Turno de trabajo (D: Día, N: Noche). Feature categórica útil para segmentar consumo por turno.'
        CODEC(ZSTD(1)),

    Origin LowCardinality(String)
        COMMENT 'Origen del suministro de combustible (ej: P068, SST, SURTIDOR-TRUCKSHOP). Permite trazabilidad de recargas.'
        CODEC(ZSTD(1)),

    Equipment LowCardinality(String)
        COMMENT 'Identificador único del camión o equipo (ej: T-210).'
        CODEC(ZSTD(1)),

    TruckFleet LowCardinality(String)
        COMMENT 'Modelo o flota del camión (ej: CAT 789C, CAT 793D). Feature categórica clave para análisis por tipo de equipo.'
        CODEC(ZSTD(1)),

    -- ⛽ Variables del sensor de combustible
    FuelLevelLiters Float32
        COMMENT 'Nivel de combustible en litros registrado por el surtidor de combustible.'
        CODEC(Gorilla, ZSTD(1)),

    FuelLevel Float32
        COMMENT 'Nivel de combustible expresado en porcentaje (0-100%). Registrado por el surtidor de combustible.'
        CODEC(Gorilla, ZSTD(1)),

    -- indexes for query optimization of the fuel supply table
    INDEX idx_equipment Equipment TYPE set(0) GRANULARITY 4,
    INDEX idx_truck_fleet TruckFleet TYPE set(0) GRANULARITY 4,
    INDEX idx_origin Origin TYPE set(0) GRANULARITY 4,
    INDEX idx_shift Shift TYPE set(0) GRANULARITY 4,
    INDEX idx_timestamp TimeStamp TYPE minmax GRANULARITY 4

)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ShiftDate)
ORDER BY (TruckFleet, Equipment, TimeStamp)
PRIMARY KEY (TruckFleet, Equipment, TimeStamp)
SETTINGS
    index_granularity = 8192,
    allow_nullable_key = 1
COMMENT 'Tabla base de suministro de combustible. Incluye información temporal, de origen, flota, equipo, nivel de combustible y un identificador secuencial único. Diseñada para análisis de consumo, trazabilidad y modelos predictivos de abastecimiento.'
"""
