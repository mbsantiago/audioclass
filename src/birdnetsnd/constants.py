from pathlib import Path

__all__ = [
    "INPUT_SAMPLES",
    "SAMPLERATE",
    "HOP_SIZE",
    "MODEL_PATH",
    "LABELS_PATH",
    "DEFAULT_THRESHOLD",
]

DEFAULT_THRESHOLD = 0.1
INPUT_SAMPLES = 144000
SAMPLERATE = 48_000
HOP_SIZE = INPUT_SAMPLES / SAMPLERATE

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
)

LABELS_PATH = (
    Path(__file__).parent / "models" / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
)
