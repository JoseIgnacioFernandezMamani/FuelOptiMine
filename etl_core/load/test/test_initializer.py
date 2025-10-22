import logging
import sys
import os
from etl_core.load.implementations import ClickHouseInitializer  # Tu nueva clase
from etl_core.load.utils.config import CH_CONFIG
from etl_core.utils.fuel_optimine_table import CREATE_TABLE_XGBOOST_FUEL

# Configurar logging para ver los mensajes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_clickhouse_initializer_connection():
    """Prueba básica de conexión usando ClickHouseInitializer"""
    print("Probando conexión básica con ClickHouseInitializer...")

    try:
        with ClickHouseInitializer() as initializer:
            # Si llegamos aquí, la conexión fue exitosa
            print("Conexión exitosa con ClickHouseInitializer!")

            # Verificar que el cliente existe
            assert initializer.client is not None, "Cliente no inicializado"

            # Hacer una consulta simple
            result = initializer.client.command("SELECT 1")
            print(f"Resultado SELECT 1: {result}")

            # Verificar configuración
            print(f"Base de datos configurada: {initializer.params['database']}")
            print(f"Host: {initializer.params['host']}")
            print(f"Puerto: {initializer.params['port']}")

        return True

    except Exception as e:
        print(f"Error en conexión básica: {e}")
        return False


def test_create_database():
    """Prueba creación de base de datos"""
    print("\nProbando creación de base de datos...")

    try:
        with ClickHouseInitializer() as initializer:
            # Crear la base de datos
            initializer.create_database()
            print("Base de datos creada/verificada exitosamente!")

            # Verificar que existe
            databases = initializer.client.query("SHOW DATABASES")
            db_names = [row[0] for row in databases.result_rows]

            if initializer.params["database"] in db_names:
                print(
                    f"Base de datos '{initializer.params['database']}' confirmada en lista"
                )
            else:
                print(
                    f"Base de datos '{initializer.params['database']}' no encontrada en: {db_names}"
                )

        return True

    except Exception as e:
        print(f"Error creando base de datos: {e}")
        return False


def test_create_table_direct():
    """Prueba creación de tabla usando DDL directo"""
    print("\nProbando creación de tabla xgboost_fuel...")

    try:
        with ClickHouseInitializer() as initializer:
            # Crear la base de datos primero
            initializer.create_database()

            # Crear tabla usando el DDL
            initializer.create_table(CREATE_TABLE_XGBOOST_FUEL)
            print("Tabla 'xgboost_fuel' creada exitosamente!")

            # Verificar que la tabla existe
            db_name = initializer.params["database"]
            tables = initializer.client.query(f"SHOW TABLES FROM {db_name}")
            table_names = [row[0] for row in tables.result_rows]

            if "xgboost_fuel" in table_names:
                print("Tabla 'xgboost_fuel' confirmada en lista")

                # Mostrar estructura de la tabla
                desc = initializer.client.query(f"DESCRIBE {db_name}.xgboost_fuel")
                print("Estructura de la tabla:")
                column_count = 0
                for row in desc.result_rows:
                    print(f"   {row[0]}: {row[1]}")
                    column_count += 1
                print(f"Total columnas: {column_count}")

                # Verificar algunas columnas clave
                column_names = [row[0] for row in desc.result_rows]
                key_columns = [
                    "Equipment",
                    "TimeStamp",
                    "TruckFleet",
                    "FuelLevelLiters",
                    "SortTimestamp",
                ]
                missing_columns = [
                    col for col in key_columns if col not in column_names
                ]

                if missing_columns:
                    print(f"Columnas clave faltantes: {missing_columns}")
                else:
                    print("Todas las columnas clave están presentes")

            else:
                print(f"Tabla 'xgboost_fuel' no encontrada en: {table_names}")

        return True

    except Exception as e:
        print(f"Error creando tabla: {e}")
        return False


def test_initialize_complete():
    """Prueba inicialización completa de la base de datos"""
    print("\nProbando inicialización completa...")

    try:
        with ClickHouseInitializer() as initializer:
            # Usar el método de inicialización completa
            initializer.initialize_database()
            print("Inicialización completa exitosa!")

            # Verificar resultado final
            db_name = initializer.params["database"]
            tables = initializer.client.query(f"SHOW TABLES FROM {db_name}")
            table_names = [row[0] for row in tables.result_rows]

            print(f"Tablas en la base de datos '{db_name}':")
            for table in table_names:
                print(f"   - {table}")

            if "xgboost_fuel" in table_names:
                print("Tabla objetivo 'xgboost_fuel' creada correctamente")
            else:
                print("ERROR: Tabla objetivo no encontrada")
                return False

        return True

    except Exception as e:
        print(f"Error en inicialización completa: {e}")
        return False


def test_table_structure():
    """Verifica la estructura detallada de la tabla"""
    print("\nVerificando estructura detallada de la tabla...")

    try:
        with ClickHouseInitializer() as initializer:
            initializer.initialize_database()

            db_name = initializer.params["database"]

            # Obtener información detallada de la tabla
            desc = initializer.client.query(f"DESCRIBE {db_name}.xgboost_fuel")

            print("Análisis de estructura de tabla:")
            sensor_cols = time_model_cols = cycle_cols = other_cols = 0

            for row in desc.result_rows:
                col_name, col_type = row[0], row[1]

                # Clasificar columnas
                if col_name in [
                    "Equipment",
                    "TimeStamp",
                    "ShiftDate",
                    "Shift",
                    "TruckFleet",
                    "FuelLevelLiters",
                    "Latitude",
                    "Longitude",
                    "Elevation",
                    "SpeedAvg",
                    "Acceleration",
                    "SlopePercent",
                    "ValidFuel",
                    "DeltaFuel",
                    "BeforeAvg",
                    "AfterAvg",
                ]:
                    sensor_cols += 1
                elif col_name in [
                    "TimeModelId",
                    "TimeStamp_tm",
                    "Status",
                    "Category",
                    "Event",
                ]:
                    time_model_cols += 1
                elif "cycle" in col_name.lower() or col_name in [
                    "CycleId",
                    "Shovel",
                    "ShovelModel",
                    "StageType",
                    "StageSequence",
                    "TimeStampIni",
                    "TimeStampFin",
                ]:
                    cycle_cols += 1
                else:
                    other_cols += 1

                # Verificar tipos nullable correctos
                if col_name.startswith(("TimeModel", "Cycle", "Shovel", "Stage")):
                    if "Nullable" not in col_type:
                        print(f"   WARNING: {col_name} debería ser Nullable")

            print(f"   Columnas sensor: {sensor_cols}")
            print(f"   Columnas time model: {time_model_cols}")
            print(f"   Columnas cycle: {cycle_cols}")
            print(f"   Otras columnas: {other_cols}")
            print(
                f"   Total: {sensor_cols + time_model_cols + cycle_cols + other_cols}"
            )

        return True

    except Exception as e:
        print(f"Error verificando estructura: {e}")
        return False


def test_error_handling():
    """Prueba manejo de errores"""
    print("\nProbando manejo de errores...")

    try:
        with ClickHouseInitializer() as initializer:
            initializer.create_database()

            # Intentar crear tabla con DDL inválido
            try:
                invalid_ddl = "CREATE TABLE invalid_syntax ( invalid )"
                initializer.create_table(invalid_ddl)
                print("ERROR: No se capturó el error esperado")
                return False
            except Exception as e:
                print(f"Error capturado correctamente: {type(e).__name__}")

        return True

    except Exception as e:
        print(f"Error inesperado en prueba de errores: {e}")
        return False


def test_context_manager():
    """Prueba el context manager"""
    print("\nProbando context manager...")

    try:
        # Verificar que se puede usar with statement
        with ClickHouseInitializer() as initializer:
            assert (
                initializer.client is not None
            ), "Cliente no inicializado en context manager"
            print("Context manager funciona correctamente!")

        print("Context manager completado correctamente")
        return True

    except Exception as e:
        print(f"Error en context manager: {e}")
        return False


def test_custom_params():
    """Prueba inicialización con parámetros personalizados"""
    print("\nProbando parámetros personalizados...")

    try:
        # Usar parámetros personalizados (pero que funcionen)
        custom_params = {"send_receive_timeout": 600}  # Timeout más largo

        with ClickHouseInitializer(**custom_params) as initializer:
            # Verificar que los parámetros se aplicaron
            assert initializer.params["send_receive_timeout"] == 600
            print("Parámetros personalizados aplicados correctamente")

            # Probar funcionalidad básica
            initializer.create_database()
            print("Funcionalidad básica con parámetros personalizados: OK")

        return True

    except Exception as e:
        print(f"Error con parámetros personalizados: {e}")
        return False


def run_all_tests():
    """Ejecuta todas las pruebas adaptadas"""
    print("Iniciando pruebas de ClickHouseInitializer simplificado")
    print("=" * 70)

    tests = [
        ("Conexión básica", test_clickhouse_initializer_connection),
        ("Creación de BD", test_create_database),
        ("Creación de tabla", test_create_table_direct),
        ("Inicialización completa", test_initialize_complete),
        ("Estructura de tabla", test_table_structure),
        ("Manejo de errores", test_error_handling),
        ("Context manager", test_context_manager),
        ("Parámetros personalizados", test_custom_params),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Excepción no controlada en {test_name}: {e}")
            results.append((test_name, False))

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE PRUEBAS:")
    print("=" * 70)

    passed = 0
    for test_name, success in results:
        status = "PASÓ" if success else "FALLÓ"
        print(f"{status:10} - {test_name}")
        if success:
            passed += 1

    print(f"\nResultado final: {passed}/{len(results)} pruebas pasaron")

    if passed == len(results):
        print("Todas las pruebas pasaron exitosamente!")
        return True
    else:
        print("Algunas pruebas fallaron. Revisa los logs arriba.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
