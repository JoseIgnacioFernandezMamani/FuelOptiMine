def _find_matching_files(self, patterns: list) -> list:
    """Encuentra archivos que coincidan con los patrones"""
    return [file for pattern in patterns for file in glob(pattern)]

def _generate_file_patterns(self, truck: str, data_type: str) -> list:
    """Genera patrones de búsqueda para tipo de conjunto y año"""
    return [
        os.path.join(
            self.base_dir,
            f"train_data_{data_type}",
            f"test_*",
            f"{self.truck}_*.csv"
        )
    ]
