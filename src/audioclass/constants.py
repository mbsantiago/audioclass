from pathlib import Path

__all__ = [
    "DEFAULT_THRESHOLD",
    "DATA_DIR",
]

DEFAULT_THRESHOLD = 0.1
BATCH_SIZE = 4

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
