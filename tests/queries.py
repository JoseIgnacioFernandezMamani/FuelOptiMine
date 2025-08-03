# queries.py

fuel_level_timeseries = """
SELECT 
    toDate(TimeStamp) AS day,
    avg(FuelLevelLiters) AS avg_fuel_level,
    min(FuelLevelLiters) AS min_fuel_level,
    max(FuelLevelLiters) AS max_fuel_level,
    count(*) AS count
FROM fuel_optimine.sensor
GROUP BY day
ORDER BY day ASC
"""

fuel_level_by_day = """
SELECT 
    TimeStamp,
    FuelLevelLiters
FROM fuel_optimine.sensor
WHERE toDate(TimeStamp) = toDate(now())
ORDER BY TimeStamp
"""
