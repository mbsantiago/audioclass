from pathlib import Path
from typing import List, Optional, Union

from soundevent import data
from tflite_runtime.interpreter import Interpreter

from audioclass.constants import DEFAULT_THRESHOLD
from audioclass.models.tflite import (
    Signature,
    TFLiteModel,
    load_model,
)
from audioclass.utils import load_artifact

__all__ = ["BirdNET"]

INPUT_SAMPLES = 144000
SAMPLERATE = 48_000
MODEL_PATH = "https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_PATH = "https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt"


class BirdNET(TFLiteModel):
    @classmethod
    def load(
        cls,
        model_path: Union[Path, str] = MODEL_PATH,
        labels_path: Union[Path, str] = LABELS_PATH,
        num_threads: Optional[int] = None,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        samplerate: int = SAMPLERATE,
        name: str = "BirdNET",
        common_name: bool = False,
    ) -> "BirdNET":
        model_path = load_artifact(model_path)
        labels_path = load_artifact(labels_path)
        interpreter = load_model(model_path, num_threads)
        tags = load_tags(labels_path, common_name=common_name)
        return cls(
            interpreter=interpreter,
            signature=get_signature(interpreter),
            tags=tags,
            confidence_threshold=confidence_threshold,
            samplerate=samplerate,
            name=name,
        )


def load_tags(
    path: Union[Path, str] = LABELS_PATH,
    common_name: bool = False,
) -> List[data.Tag]:
    """Load BirdNET labels from a file.

    Parameters
    ----------
    path : Path
        Path to the file containing the labels.
    common_name : bool, optional
        Whether to return the common name instead of the scientific name, by
        default False

    Returns
    -------
    List[str]
        List of labels.

    Notes
    -----
    The file should contain one label per line and should be in the format:

    ```
    <scientific_name>_<common_name>
    ```
    """
    path = load_artifact(path)

    with open(path) as f:
        labels = f.read().splitlines()

    index = 1 if common_name else 0
    key = "common_name" if common_name else "scientific_name"
    return [
        data.Tag(key=key, value=label.split("_")[index]) for label in labels
    ]


def get_signature(interpreter: Interpreter) -> Signature:
    """Get the signature of a BirdNET model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not len(input_details) == 1:
        raise ValueError("Model must have exactly one input tensor")

    if not len(output_details) == 1:
        raise ValueError("Model must have exactly one output tensor")

    input_details = input_details[0]
    output_details = output_details[0]

    if not len(input_details["shape"]) == 2:
        raise ValueError("Input tensor must have 2 dimensions")

    return Signature(
        input_index=input_details["index"],
        input_length=input_details["shape"][-1],
        input_dtype=input_details["dtype"],
        classification_index=output_details["index"],
        feature_index=output_details["index"] - 1,
    )
