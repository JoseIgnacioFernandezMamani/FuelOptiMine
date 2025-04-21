from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Función para validar la conexión
def test_connection(engine):
    try:
        with engine.connect() as connection:  # Abre la conexión de manera segura
            result = connection.execute(text("SELECT 1"))  # Ejecuta una consulta de prueba
            test = result.fetchone()[0] # obtiene el primner registro
            if test == 1:
                print(f"Succeful connection, test: {test}")
    except SQLAlchemyError as e:
        print(f"Error en la conexión: {e}")
        
# configuracion de la conexion
server = '172.16.5.204'
database = 'jmineops'
driver = 'ODBC Driver 17 for SQL Server'

# cadena de conexion
connection_string = f'mssql+pyodbc://@{server}/{database}?driver={driver}&TrustServerCertificate=yes&trusted_connection=yes'

# crear el nucleo de conexion con la base de datos
engine = create_engine(connection_string)

# probando
test_connection(engine)


# carga selectiva de 1000 datos de prueba

from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

# configuraciones de la conexion
server = 'mscexagonrep'
database = 'jmineops'
driver = 'ODBC Driver 17 for SQL Server'
# cadena de conexion
connection_string = f'mssql+pyodbc://@{server}/{database}?driver={driver}&TrustServerCertificate=yes&trusted_connection=yes'

# crear el nucleo de conexion con la base de datos
engine = create_engine(connection_string)

try:
    query = '''
    SELECT TOP (100000)
        * 
    FROM 
        [Msc_Dev].[rep].[custom_Fuel_History]
    WHERE 
        FuelLevelTimestamp BETWEEN '2024-01-01 08:07:30' AND '2025-02-28 23:59:59
    ORDER BY '
    '''
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        print(df.tail(40))
except SQLAlchemyError as e:
    print(f"Error en la conexión: {e}")









