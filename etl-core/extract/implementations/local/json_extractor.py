import json

class JSONExtractor(BaseExtractor):
    def __init__(self, truck: str, data_dir: Path):
        super().__init__(truck, data_dir, file_extension="json")

    def _load_single_file(self, file_path: str, data_type: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error en {Path(file_path).name}: {str(e)}")
            return {}

    def load_data(self) -> Dict[str, Dict]:
        datasets = {}
        for data_type in ["sensor", "time_model", "cycle"]:  # Ejemplo
            patterns = self._generate_file_patterns(data_type)
            files = self._find_matching_files(patterns)
            
            if not files:
                print(f"Advertencia: No hay archivos para {data_type}")
                datasets[data_type] = {}
                continue
            
            with ThreadPoolExecutor() as executor:
                dfs = list(executor.map(lambda f: self._load_single_file(f, data_type), files))
            
            combined_data = {}
            for data in dfs:
                combined_data.update(data)
            
            datasets[data_type] = combined_data
        
        return datasets