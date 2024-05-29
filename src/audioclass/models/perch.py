from pathlib import Path
from typing import List, Union

import tensorflow_hub as hub
from soundevent import data
from tensorflow.python.types.core import Callable

from audioclass.constants import DATA_DIR, DEFAULT_THRESHOLD
from audioclass.models.tensorflow import (
    Signature,
    TensorflowModel,
)
from audioclass.utils import load_artifact

SAMPLERATE = 32_000
INPUT_SAMPLES = 160_000
MODEL_PATH = "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4"
TAGS_PATH = DATA_DIR / "perch" / "label.csv"


class Perch(TensorflowModel):
    @classmethod
    def load(
        cls,
        model_url: Union[Path, str] = MODEL_PATH,
        tags_url: Union[Path, str] = TAGS_PATH,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        samplerate: int = SAMPLERATE,
        name: str = "Perch",
    ):
        model = hub.load(model_url)
        tags = load_tags(tags_url)
        callable = model.signatures["serving_default"]  # type: ignore
        signature = get_signature(callable)
        return cls(
            callable,
            signature,
            tags,
            confidence_threshold=confidence_threshold,
            samplerate=samplerate,
            name=name,
        )


def get_signature(callable: Callable) -> Signature:
    """Get the signature of a Perch model."""
    function_type = callable.function_type
    parameters = function_type.parameters

    if not len(parameters) == 1:
        raise ValueError("Model must have exactly one input tensor")

    input_name = next(iter(parameters))
    input_param = parameters[input_name]
    input_annotations = input_param.annotation

    if len(input_annotations.shape) != 2:
        raise ValueError("Input tensor must have 2 dimensions")

    output = function_type.return_annotation.mapping
    if len(output) != 2:
        raise ValueError("Model must have exactly two output tensors")

    # NOTE: Here we assume that the first output tensor is the feature tensor
    # and the second output tensor is the classification tensor
    feature_name, classification_name = output

    return Signature(
        input_name=input_name,
        classification_name=classification_name,
        feature_name=feature_name,
        input_length=input_annotations.shape[1],
        input_dtype=input_annotations.dtype.as_numpy_dtype,
    )


def load_tags(path: Union[Path, str] = TAGS_PATH) -> List[data.Tag]:
    path = load_artifact(path)
    tags = path.read_text().splitlines()[1:]
    return [data.Tag(key="ebird2021", value=tag) for tag in tags]
