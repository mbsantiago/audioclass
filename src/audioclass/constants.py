"""Module containing constants for the audioclass package."""

from pathlib import Path

__all__ = [
    "DEFAULT_THRESHOLD",
    "DATA_DIR",
]

DEFAULT_THRESHOLD: float = 0.1
"""Default confidence threshold for model predictions.

This value is used as the minimum probability for a prediction to be considered
valid.
"""

BATCH_SIZE: int = 4
"""Default size of batch for model inference.

This value determines the number of audio clips that are processed together in
a single batch.
"""

ROOT_DIR: Path = Path(__file__).parent
"""Root directory of the module.

This is the directory where the `audioclass` module is located.
"""

DATA_DIR: Path = ROOT_DIR / "data"
"""Directory containing supporting data files.

This is the directory where any additional data files required by the
`audioclass` package are stored.
"""
