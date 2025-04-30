def _detect_separator(self, file_path: str) -> str:
    """Detecta el separador leyendo la primera línea"""
    separadores = [';', '\t', ',']
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            primera_linea = f.readline().strip()
        return next((s for s in separadores if s in primera_linea), ',')
    except:
        return ','
