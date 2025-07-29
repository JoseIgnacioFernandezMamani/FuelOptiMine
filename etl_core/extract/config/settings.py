from pathlib import Path

# DATA_DIR = Path(os.path.abspath(os.path.join("data-set")))
DATA_DIR: Path = Path(__file__).parent.parent.parent.parent / "data-set"
