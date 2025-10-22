# Guía Completa de Instalación y Configuración Segura

## PASO 1: CLICKHOUSE - Instalación y Configuración Completa

### 1.1 Instalación de ClickHouse

> editar las variables de entorno .env segun vean convenientes, basado en estas instrucciones

```bash
sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt-get -y update
sudo apt-get install clickhouse-server=25.6.5.41 clickhouse-client=25.6.5.41 # versiones 25.6.5.41 usadas en el proyecto, asegurese de tener una version similar a 25.*.*.*
# poner contrasenia para usuario por defecto= "msc_admin".

# si no tiene systemctl instalar con apt-get install -y systemctl
# puedes requerir configurar la zona y region
# configurar enlace simbolico de la zona horaria de la zona que sea 
# sudo ln -fs /usr/share/zoneinfo/America/La_Paz /etc/localtime
# reconfigurar el tzdata en modo no interactivo
# sudo DEBIAN_FRONTEND=noninteractive dpkg-reconfigure tzdata

# establecer la codificacion utf-8 por defecto
# abrir la codificaion con sudo nano /etc/default/locale
# agregar esto al final: LANG=es_BO.UTF-8
# verifica con el comando despues de reiniciar:  locale -a | grep en_US

# reiniciar el sistema con: sudo reboot

sudo systemctl enable clickhouse-server 
sudo systemctl start clickhouse-server
# opcionalmente verificar si funciona con sudo systemctl status clickhouse-server
```

### 1.2 Configurar Contraseña Usuario Default (opcional si no coloco una contrasenia despues de la instalacion)

```bash
# instalar un editor de texto de no tener uno con
# sudo apt-get update
# sudo apt-get install -y nano

sudo nano /etc/clickhouse-server/users.xml
```

Buscar `<users>` y dentro agregar:

```xml
<users>
    <default>
        <password>msc_password</password>
        <networks>
            <ip>::/0</ip>
        </networks>
        <profile>default</profile>
        <quota>default</quota>
    </default>
</users>
```

```bash
sudo systemctl restart clickhouse-server
```

### 1.3 Crear Base de Datos

```bash
clickhouse-client #coloque la contrasenia password, podria estar el texto oculto, escriba igualment la contrasenia especificada tal cual y enter.
```

```sql
# esta dentro de la shell de clickhouse ahora ejecute lo siguiente:
CREATE DATABASE IF NOT EXISTS fuel_optimine;
# opcionalmente ejecute SHOW DATABASES;
# si todo esta correcto podra ver la base de datos fuel_optimine
```

### 1.4 Crear Usuario con Privilegios Limitados

```sql
CREATE USER msc_user1 
    IDENTIFIED WITH sha256_password BY 'msc_user1_password'
    HOST IP '127.0.0.1', IP '::1'
    DEFAULT DATABASE fuel_optimine;
```

### 1.5 Asignar Privilegios al Usuario

```sql
GRANT CREATE TABLE, DROP TABLE, ALTER TABLE, TRUNCATE ON fuel_optimine.* TO msc_user1;
GRANT CREATE VIEW, DROP VIEW, ALTER VIEW ON fuel_optimine.* TO msc_user1;
GRANT CREATE DICTIONARY, DROP DICTIONARY ON fuel_optimine.* TO msc_user1;
GRANT SELECT, INSERT, ALTER UPDATE, ALTER DELETE ON fuel_optimine.* TO msc_user1;
GRANT CREATE TEMPORARY TABLE ON fuel_optimine.* TO msc_user1;
GRANT OPTIMIZE ON fuel_optimine.* TO msc_user1;
GRANT SHOW TABLES, SHOW COLUMNS, SHOW DICTIONARIES ON fuel_optimine.* TO msc_user1;

# opcionalmente mirar los privilegios:  SHOW GRANTS FOR msc_user1;
EXIT;
```

### 1.6 Verificar Usuario Limitado existe

```bash
clickhouse-client --user=msc_user1 --password=msc_user1_password --database=fuel_optimine
```

## PASO 2: POSTGRESQL - Instalación y Configuración Completa

### 2.1 Instalación de PostgreSQL

```bash
sudo apt-get update
sudo apt-get install -y postgresql-16
sudo systemctl enable postgresql
sudo systemctl start postgresql
sudo systemctl status postgresql
```

### 2.2 Configurar Contraseña Usuario Postgres

```bash
sudo -u postgres psql
```

```sql
ALTER USER postgres WITH PASSWORD 'admin_secure_password';

# salir con "\q"
```

### 2.3 Crear Base de Datos

```bash
sudo -u postgres psql
```

```sql
CREATE DATABASE fuel_optimine
    WITH 
    ENCODING = 'UTF8'
    TEMPLATE = template0;

# listar todas las bases de datos con :\l
# conectarte con una base de datos con:\c fuel_optimine
```

### 2.4 Crear Usuario con Privilegios Limitados

```sql
CREATE USER msc_user2_admin WITH PASSWORD 'msc_user2_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOREPLICATION
    LOGIN;

# listar usuarios y roles con:\du
```

### 2.5 Revocar Privilegios Públicos


```sql
# todo usuario creados pertenecen al esquema publico, por defecto este esquema tiene permisos de alto nivel, algo peligrosos
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE fuel_optimine FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
```

### 2.6 Asignar Privilegios al Usuario

```sql
GRANT CONNECT ON DATABASE fuel_optimine TO msc_user2_admin;
GRANT USAGE ON SCHEMA public TO msc_user2_admin;
GRANT CREATE ON SCHEMA public TO msc_user2_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO msc_user2_admin;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO msc_user2_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO msc_user2_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO msc_user2_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO msc_user2_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT EXECUTE ON FUNCTIONS TO msc_user2_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT EXECUTE ON ROUTINES TO msc_user2_admin;

# para ver los resultados ejecutar: 
SELECT 
    *
FROM pg_roles
WHERE rolname = 'msc_user2_admin';
\q
```

### 2.7 Verificar Usuario Limitado

```bash
psql -U msc_user2_admin -d fuel_optimine -h localhost
```

Introducir contraseña: `msc_user2_password`

### 2.8 Configuración de Acceso Remoto

```bash
# Editar postgresql.conf
sudo nano /etc/postgresql/16/main/postgresql.conf

# modificar en el archivo la siguiente linea para permitir conexiones desde cualquier IP 
listen_addresses = '*'  # o 'localhost,192.168.1.100'
```

```bash
# Editar pg_hba.conf para reglas de autenticación
sudo nano /etc/postgresql/16/main/pg_hba.conf

# agregar a final para permitir cualquier usuario de cualquier ip
host    fuel_optimine    msc_user2_admin    127.0.0.1/32    scram-sha-256
host    fuel_optimine    msc_user2_admin    ::1/128         scram-sha-256

# Reiniciar PostgreSQL
sudo systemctl restart postgresql
```

## PASO 3: Entorno de desarrollo de python

```bash
# opcional, crear y activas variable de entorno
python3 -m venv fueloptimine-env
source fueloptimine-env/bin/activate 


```

```bash
# instalar dependencias necesarias
pip install -r requirements.txt

# dirigirse al directorio del proyecto "FuelOptiMine" y dar permiso a los scripts
chmod +x mlflow_server/start_mlflow.sh
chmod +x mlflow_server/stop_mlflow.sh

# ejecurtar el servidor de mlflow, se tiene opcionalmente en el mismo directorio un script para detener el sevidor
mlflow_server/start_mlflow.sh

        

```
