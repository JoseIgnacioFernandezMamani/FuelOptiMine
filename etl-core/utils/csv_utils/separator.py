def detect_separator(file_path: str) -> str:
    """
    Detects the separator by reading the first line
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        str: Detected separator
    """
    separadores = [';', '\t', ',']
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            primera_linea = f.readline().strip()
        return next((s for s in separadores if s in primera_linea), ',')
    except:
        return ','