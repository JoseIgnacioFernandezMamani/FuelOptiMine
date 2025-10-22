CREATE_TABLE_XGBOOST_FUEL = """
-- =================================================================
-- Tabla para entrenamiento de modelo XGBoost de predicción de consumo de combustible
-- =================================================================

CREATE TABLE IF NOT EXISTS fuel_optimine.xgboost_fuel
(
    -- key fields
    Equipment LowCardinality(String) 
        COMMENT 'Identificador único del equipo/camión (ej: T-210, T-215). Feature categórica principal para segmentación del modelo'
        CODEC(ZSTD(1)),
        
    TimeStamp DateTime64(3, 'UTC') 
        COMMENT 'Marca temporal UTC del registro del sensor. Variable temporal fundamental para análisis de series de tiempo y patrones operacionales'
        CODEC(Delta(8), ZSTD(1)),
        
    ShiftDate Date 
        COMMENT 'Fecha del turno de trabajo (ej: 2023-01-01).'
        CODEC(Delta(2), ZSTD(1)),
   
    -- common fields
    Shift LowCardinality(String) 
        COMMENT 'Turno de trabajo (ej: D, N).'
        CODEC(ZSTD(1)),
        
    TruckFleet LowCardinality(String) 
        COMMENT 'Flota a la que pertenece el camión. Feature categórica para agrupar por modelo de equipo (ej: T-210).'
        CODEC(ZSTD(1)),
   
    -- sensor fields
    FuelLevelLiters Float32 
        COMMENT 'Nivel de combustible en litros medido por sensor. Variable objetivo principal para predicciones de consumo'
        CODEC(Gorilla, ZSTD(1)),
        
    Latitude Float64 
        COMMENT 'Coordenada geográfica latitud (grados decimales).'
        CODEC(Gorilla, ZSTD(1)),
        
    Longitude Float64 
        COMMENT 'Coordenada geográfica longitud (grados decimales).'
        CODEC(Gorilla, ZSTD(1)),
        
    Elevation Float32 
        COMMENT 'Elevación/altitud en metros sobre nivel del mar. Análisis de impacto altura en consumo de combustible'
        CODEC(Gorilla, ZSTD(1)),
        
    SpeedAvg Float32 
        COMMENT 'Velocidad promedio en km/h durante el período de registro. Feature predictora fundamental para consumo energético'
        CODEC(Gorilla, ZSTD(1)),
        
    Acceleration Float32 
        COMMENT 'Aceleración en m/s² (positiva=acelerando, negativa=desacelerando). Indicador de agresividad en conducción'
        CODEC(Gorilla, ZSTD(1)),
        
    SlopePercent Float32 
        COMMENT 'Pendiente del terreno en porcentaje (positiva=subida, negativa=bajada).'
        CODEC(Gorilla, ZSTD(1)),
        
    ValidFuel Float32 
        COMMENT 'Indicador del nivel de combustible aproximado cuando termina el recargado de combustible de un camión.'
        CODEC(Gorilla, ZSTD(1)),
        
    DeltaFuel Float32 
        COMMENT 'Cambio en nivel de combustible respecto al registro anterior (litros). Indicador directo de consumo o recarga en un evento de recarga'
        CODEC(Gorilla, ZSTD(1)),
        
    BeforeAvg Float32 
        COMMENT 'Promedio de consumo en ventana temporal anterior del evento de recarga. Feature de validación'
        CODEC(Gorilla, ZSTD(1)),
        
    AfterAvg Float32 
        COMMENT 'Promedio de consumo en ventana temporal posterior del evento de recarga. Feature de validación'
        CODEC(Gorilla, ZSTD(1)),
        
    -- time model fields
    TimeModelId Nullable(String) 
        COMMENT 'Identificador único de un evento en modelo de tiempo.'
        CODEC(ZSTD(1)),
        
    TimeStamp_tm Nullable(DateTime64(3, 'UTC')) 
        COMMENT 'Timestamp específico del modelo de tiempo (puede diferir del sensor). Referencia temporal para eventos del modelo de tiempos'
        CODEC(Delta(8), ZSTD(1)),
        
    Status Nullable(String) 
        COMMENT 'Estado operacional del equipo (DEMORA/MANTENIMIENTO/OPERATIVO/RESERVA).'
        CODEC(ZSTD(1)),
        
    Category Nullable(String) 
        COMMENT 'Categoría de actividad operacional (D_NO_PROGRAMADA/D_PROGRAMADA/EFECTIVO/M_NO_PROGRAMADO/M_PROGRAMADO/REPARACION/RESERVA). Dato que enriquece la categoria de cada estado de un camion.'
        CODEC(ZSTD(1)),
        
    Event Nullable(String) 
        COMMENT 'Tipo de evento operacional específico. Feature categórica para análisis detallado de actividades (ej: ALMUERZO/CENA, FALTA DE COMBUSTIBLE, TRASLADO, ETC).'
        CODEC(ZSTD(1)),
   
    -- =================================================================
    -- 🚛 DATOS DE CICLO OPERACIONAL
    -- =================================================================
    CycleId Nullable(String) 
        COMMENT 'Identificador único de un ciclo operacional completo.'
        CODEC(ZSTD(1)),
        
    Shovel Nullable(String) 
        COMMENT 'Identificador de la pala cargadora utilizada durante el ciclo de carguio y acarreo. Feature categórica para análisis de eficiencia por tipo de pala'
        CODEC(ZSTD(1)),
        
    ShovelModel Nullable(String) 
        COMMENT 'Modelo de la pala cargadora utilizada en el ciclo. Feature categórica para análisis de eficiencia por tipo de pala'
        CODEC(ZSTD(1)),
        
    StageType Nullable(String) 
        COMMENT 'Tipo de etapa del ciclo (empty_traveling/load_positioning/load_travel/load_waiting_queue/material_loading/material_unloading/spotting_empty/waiting_empty). Feature fundamental para segmentación de patrones operacionales'
        CODEC(ZSTD(1)),
        
    StageSequence Nullable(UInt8) 
        COMMENT 'Secuencia numérica de la etapa del ciclo (1=empty_traveling, 2=load_positioning, etc.). Feature ordinal para análisis temporal del ciclo'
        CODEC(ZSTD(1)),
        
    TimeStampIni Nullable(DateTime64(3, 'UTC')) 
        COMMENT 'Timestamp de inicio de una etapa del ciclo. Permite cálculo de duración'
        CODEC(Delta(8), ZSTD(1)),
        
    TimeStampFin Nullable(DateTime64(3, 'UTC')) 
        COMMENT 'Timestamp de fin de una etapa del ciclo. Permite cálculo de duración'
        CODEC(Delta(8), ZSTD(1)),
        
    LoadingZone Nullable(String) 
        COMMENT 'Zona de carga donde se realizó la actividad de etapa 4.'
        CODEC(ZSTD(1)),
        
    Material Nullable(String) 
        COMMENT 'Tipo de material transportado (mineral, estéril, etc.) solo aparece en etapa 8. Feature categórica que impacta significativamente el consumo?'
        CODEC(ZSTD(1)),
        
    MeasuredTonnage Nullable(Float32) 
        COMMENT 'Tonelaje real medido por sensores de peso, solo aparece en la etapa 8. Feature numérica clave para predicción de consumo basado en carga'
        CODEC(Gorilla, ZSTD(1)),
        
    ReportedTonnage Nullable(Float32) 
        COMMENT 'Tonelaje reportado por los operadores, solo aparece en la etapa 8. Feature numérica clave para predicción de consumo basado en carga'
        CODEC(Gorilla, ZSTD(1)),
        
    DestinationType Nullable(String) 
        COMMENT 'Tipo de destino (Botadero,Stockpile, etc), solo aparece en la etapa 8. Feature categórica para análisis de tipo de destino y su impacto en consumo'
        CODEC(ZSTD(1)),
        
    Destination Nullable(String) 
        COMMENT 'Destino específico del material, solo aparece en la etapa 8. Feature detallada para análisis de rutas óptimas y consumo por destino'
        CODEC(ZSTD(1)),
        
    Distance Nullable(Float32) 
        COMMENT 'Distancia recorrida en el ciclo (m). Feature numérica fundamental para medir el impacto del consumo por distancia'
        CODEC(Gorilla, ZSTD(1)),
        
    Latitude_cycle Nullable(Float64) 
        COMMENT 'Coordenada latitud promedio del ciclo, en grados. Feature geoespacial.'
        CODEC(Gorilla, ZSTD(1)),
        
    Longitude_cycle Nullable(Float64) 
        COMMENT 'Coordenada longitud promedio del ciclo, en grados. Feature geoespacial.'
        CODEC(Gorilla, ZSTD(1)),
        
    Elevation_cycle Nullable(Float32) 
        COMMENT 'Elevación promedio del ciclo en metros. Feature topográfica para análisis de impacto de altitud en consumo por ciclo'
        CODEC(Gorilla, ZSTD(1)),
        
    TimeEfficiencyPercentage Nullable(Float32) 
        COMMENT 'Porcentaje de eficiencia de tiempo de cada etapa del ciclo, toma en cuenta cuanto del tiempo uso esta etapa respecto al tiempo total del ciclo.'
        CODEC(Gorilla, ZSTD(1)),
   
    -- unified sorting timestamp
    SortTimestamp DateTime64(3, 'UTC') 
        COMMENT 'Timestamp unificado para ordenamiento de registros, toma el registro de tiempo minimo entre TimeStamp de sensores, TimeStampInicio y TimeStampFin de cyclos y TimeStamp_tm del modelo de tiempos.'
        CODEC(Delta(8), ZSTD(1)),

    -- Indexes for query optimization of the xgboost model
    INDEX idx_equipment Equipment TYPE set(0) GRANULARITY 4,
    INDEX idx_truck_fleet TruckFleet TYPE set(0) GRANULARITY 4,
    INDEX idx_time_model_id TimeModelId TYPE bloom_filter GRANULARITY 2,
    INDEX idx_status Status TYPE bloom_filter GRANULARITY 2,
    INDEX idx_cycle_id CycleId TYPE bloom_filter GRANULARITY 2,
    INDEX idx_stage_sequence StageSequence TYPE set(0) GRANULARITY 2,
    INDEX idx_sort_timestamp SortTimestamp TYPE minmax GRANULARITY 4
   
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(TimeStamp)
ORDER BY (TruckFleet, Equipment, SortTimestamp)
PRIMARY KEY (TruckFleet, Equipment, SortTimestamp)
SETTINGS
    index_granularity = 8192,
    allow_nullable_key = 1
COMMENT 'Tabla unificada para entrenamiento de modelo XGBoost de predicción de consumo de combustible en camiones mineros. Combina datos de sensores en tiempo real, modelos de tiempo operacional y ciclos para análisis predictivo de consumo de combustible. Diseñada específicamente para machine learning'
"""
