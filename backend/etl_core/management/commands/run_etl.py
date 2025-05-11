# etl_core/management/commands/run_etl.py
from django.core.management.base import BaseCommand
from etl_core.etl.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.models import FuelSupply, Cycle, Sensor, TimeModel


class Command(BaseCommand):  # type: ignore
    help = "Ejecuta el ETL y guarda los datos en la base de datos"

    def add_arguments(self, parser):
        parser.add_argument("--truck", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument(
            "--base_dir",
            type=str,
            default="etl_core/etl/extract/implementations/local/datasets",
        )

    def handle(self, *args, **options):
        truck_id = options["truck"]
        dataset = options["dataset"]

        # 1. Extraer datos
        extractor = CSVExtractor(dataset=dataset, truck=truck_id)
        datasets, errors = extractor.load_data()

        # 2. Guardar en Django
        for data_type, df in datasets.items():
            for row in df.to_dicts():
                if data_type == "fuel_supply":
                    FuelSupply.objects.create(**row)

        self.stdout.write(self.style.SUCCESS("✅ Datos cargados exitosamente"))
