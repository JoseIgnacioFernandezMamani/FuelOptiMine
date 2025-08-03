#!/bin/bash

# Configuración de la base de datos y usuario
CLICKHOUSE_DB="fuel_analytics"
CLICKHOUSE_USER="fuel_user"
CLICKHOUSE_PASSWORD="S3cr3tP@ss"

# Comandos SQL para inicialización
clickhouse-client --user default --password "password" <<EOSQL
-- Crear base de datos si no existe
CREATE DATABASE IF NOT EXISTS ${CLICKHOUSE_DB};

-- Crear usuario dedicado
CREATE USER IF NOT EXISTS ${CLICKHOUSE_USER} IDENTIFIED BY '${CLICKHOUSE_PASSWORD}';

-- Otorgar permisos
GRANT ALL ON ${CLICKHOUSE_DB}.* TO ${CLICKHOUSE_USER} WITH GRANT OPTION;

-- Mostrar resultados (opcional para debug)
SHOW DATABASES;
SHOW USERS;
EOSQL

echo "✅ Base de datos '${CLICKHOUSE_DB}' y usuario '${CLICKHOUSE_USER}' configurados en ClickHouse."