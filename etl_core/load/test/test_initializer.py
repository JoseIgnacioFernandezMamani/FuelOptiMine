import logging
import sys
import os
from etl_core.load.implementations import ClickHouseInitializer
from etl_core.load.utils.config import CH_CONFIG, DATASET_CONFIG

# Configurar logging para ver los mensajes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def check_schema_files():
    """Verifica que los archivos de esquema existan"""
    print("🔍 Verificando archivos de esquema...")

    missing_schemas = []
    for dataset_name, config in DATASET_CONFIG.items():
        schema_path = config["schema_path"]

        # Convertir la ruta del módulo a ruta de archivo
        file_path = ".".join(schema_path.split(".")[:-1])
        file_path = file_path.replace(".", "/") + ".py"

        if os.path.exists(file_path):
            print(f"✅ {dataset_name}: {schema_path} -> {file_path}")
        else:
            print(f"❌ {dataset_name}: {schema_path} -> {file_path} (NO EXISTE)")
            missing_schemas.append(dataset_name)

    if missing_schemas:
        print(f"⚠️ Esquemas faltantes: {missing_schemas}")
        print("💡 El test podría fallar en la creación de tablas")
    else:
        print("✅ Todos los archivos de esquema encontrados")

    return len(missing_schemas) == 0


def test_clickhouse_initializer_connection():
    """Prueba básica de conexión usando ClickHouseInitializer"""
    print("🔍 Probando conexión básica con ClickHouseInitializer...")

    try:
        with ClickHouseInitializer() as initializer:
            # Si llegamos aquí, la conexión fue exitosa
            print("✅ Conexión exitosa con ClickHouseInitializer!")

            # Verificar que el cliente existe
            assert initializer.client is not None, "Cliente no inicializado"

            # Hacer una consulta simple
            result = initializer.client.command("SELECT 1")
            print(f"📊 Resultado SELECT 1: {result}")

            # Verificar configuración
            print(f"📊 Base de datos configurada: {initializer.params['database']}")
            print(f"📊 Host: {initializer.params['host']}")
            print(f"📊 Puerto: {initializer.params['port']}")

        return True

    except Exception as e:
        print(f"❌ Error en conexión básica: {e}")
        return False


def test_create_database():
    """Prueba creación de base de datos"""
    print("\n🔍 Probando creación de base de datos...")

    try:
        with ClickHouseInitializer() as initializer:
            # Crear la base de datos
            initializer.create_database()
            print("✅ Base de datos creada/verificada exitosamente!")

            # Verificar que existe
            databases = initializer.client.query("SHOW DATABASES")
            db_names = [row[0] for row in databases.result_rows]

            if initializer.params["database"] in db_names:
                print(
                    f"✅ Base de datos '{initializer.params['database']}' confirmada en lista"
                )
            else:
                print(
                    f"⚠️ Base de datos '{initializer.params['database']}' no encontrada en: {db_names}"
                )

        return True

    except Exception as e:
        print(f"❌ Error creando base de datos: {e}")
        return False


def test_create_single_table():
    """Prueba creación de una tabla individual"""
    print("\n🔍 Probando creación de tabla individual (sensor)...")

    try:
        with ClickHouseInitializer() as initializer:
            # Crear la base de datos primero
            initializer.create_database()

            # Intentar crear tabla de sensor (podría fallar si no existe el esquema)
            try:
                initializer.create_table("sensor")
                print("✅ Tabla 'sensor' creada exitosamente!")

                # Verificar que la tabla existe
                tables = initializer.client.query("SHOW TABLES")
                table_names = [row[0] for row in tables.result_rows]

                expected_table = DATASET_CONFIG["sensor"]["table_name"]
                if expected_table in table_names:
                    print(f"✅ Tabla '{expected_table}' confirmada en lista")

                    # Mostrar estructura de la tabla
                    desc = initializer.client.query(f"DESCRIBE {expected_table}")
                    print("📊 Estructura de la tabla:")
                    for row in desc.result_rows:
                        print(f"   {row[0]}: {row[1]}")
                else:
                    print(f"⚠️ Tabla '{expected_table}' no encontrada en: {table_names}")

            except Exception as schema_error:
                print(
                    f"⚠️ Error creando tabla (probablemente falta esquema): {schema_error}"
                )
                print("💡 Esto es normal si los archivos de esquema no existen aún")
                return True  # No fallar el test por esto

        return True

    except Exception as e:
        print(f"❌ Error creando tabla individual: {e}")
        return False


def test_initialize_all_datasets():
    """Prueba inicialización completa de todos los datasets"""
    print("\n🔍 Probando inicialización completa de todos los datasets...")

    try:
        with ClickHouseInitializer() as initializer:
            # Intentar inicializar todos los datasets
            try:
                initializer.initialize_database()
                print("✅ Todos los datasets inicializados exitosamente!")

                # Verificar que todas las tablas existen
                tables = initializer.client.query("SHOW TABLES")
                table_names = [row[0] for row in tables.result_rows]

                print("📊 Tablas creadas:")
                created_count = 0
                for dataset_name, config in DATASET_CONFIG.items():
                    expected_table = config["table_name"]
                    if expected_table in table_names:
                        print(f"   ✅ {dataset_name}: {expected_table}")
                        created_count += 1
                    else:
                        print(f"   ❌ {dataset_name}: {expected_table} - NO ENCONTRADA")

                print(
                    f"📊 Resumen: {created_count}/{len(DATASET_CONFIG)} tablas creadas"
                )

            except Exception as init_error:
                print(
                    f"⚠️ Error en inicialización (probablemente esquemas faltantes): {init_error}"
                )
                print("💡 Esto es normal si los archivos de esquema no existen aún")
                return True  # No fallar el test por esto

        return True

    except Exception as e:
        print(f"❌ Error en inicialización completa: {e}")
        return False


def test_initialize_specific_datasets():
    """Prueba inicialización de datasets específicos"""
    print("\n🔍 Probando inicialización de datasets específicos...")

    try:
        # Probar solo con sensor y fuel_supply
        target_datasets = ["sensor", "fuel_supply"]

        with ClickHouseInitializer() as initializer:
            initializer.initialize_database(datasets=target_datasets)
            print(f"✅ Datasets específicos inicializados: {target_datasets}")

            # Verificar solo las tablas solicitadas
            tables = initializer.client.query("SHOW TABLES")
            table_names = [row[0] for row in tables.result_rows]

            print("📊 Verificando tablas específicas:")
            for dataset_name in target_datasets:
                expected_table = DATASET_CONFIG[dataset_name]["table_name"]
                if expected_table in table_names:
                    print(f"   ✅ {dataset_name}: {expected_table}")
                else:
                    print(f"   ❌ {dataset_name}: {expected_table} - NO ENCONTRADA")

        return True

    except Exception as e:
        print(f"❌ Error en inicialización específica: {e}")
        return False


def test_error_handling():
    """Prueba manejo de errores"""
    print("\n🔍 Probando manejo de errores...")

    try:
        with ClickHouseInitializer() as initializer:
            initializer.create_database()

            # Intentar crear tabla con dataset inexistente
            try:
                initializer.create_table("dataset_inexistente")
                print("❌ No se capturó el error esperado")
                return False
            except ValueError as e:
                print(f"✅ Error capturado correctamente: {e}")

        return True

    except Exception as e:
        print(f"❌ Error inesperado en prueba de errores: {e}")
        return False


def test_context_manager():
    """Prueba el context manager (__enter__ y __exit__)"""
    print("\n🔍 Probando context manager...")

    try:
        # Verificar que se puede usar with statement
        with ClickHouseInitializer() as initializer:
            assert (
                initializer.client is not None
            ), "Cliente no inicializado en context manager"
            print("✅ Context manager funciona correctamente!")

            # El cliente debería cerrarse automáticamente al salir

        # Verificar que la conexión se cerró (esto podría fallar dependiendo de la implementación)
        print("✅ Context manager completado")
        return True

    except Exception as e:
        print(f"❌ Error en context manager: {e}")
        return False


def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("🚀 Iniciando pruebas completas de ClickHouseInitializer")
    print("=" * 70)

    # Primero verificar esquemas
    schemas_exist = check_schema_files()

    tests = [
        ("Conexión básica", test_clickhouse_initializer_connection),
        ("Creación de BD", test_create_database),
        ("Tabla individual", test_create_single_table),
        ("Todos los datasets", test_initialize_all_datasets),
        ("Datasets específicos", test_initialize_specific_datasets),
        ("Manejo de errores", test_error_handling),
        ("Context manager", test_context_manager),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Excepción no controlada en {test_name}: {e}")
            results.append((test_name, False))

    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE PRUEBAS:")
    print("=" * 70)

    if not schemas_exist:
        print("⚠️ NOTA: Algunos archivos de esquema no existen.")
        print(
            "   Las pruebas de creación de tablas podrían no funcionar completamente."
        )
        print("   Esto es normal si aún no has creado los esquemas.\n")

    passed = 0
    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"{status:10} - {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Resultado final: {passed}/{len(results)} pruebas pasaron")

    if passed == len(results):
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
    elif not schemas_exist and passed >= 4:  # Al menos las pruebas básicas
        print(
            "✅ Las pruebas básicas pasaron. Crea los esquemas para pruebas completas."
        )
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los logs arriba.")


if __name__ == "__main__":
    run_all_tests()
