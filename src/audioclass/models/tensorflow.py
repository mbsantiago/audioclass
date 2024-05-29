from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import DTypeLike
from soundevent import data
from tensorflow.python.types.core import Callable

from audioclass.models.base import ClipClassificationModel, ModelOutput
from audioclass.utils import flat_sigmoid

__all__ = [
    "TensorflowModel",
    "Signature",
]


@dataclass
class Signature:
    input_name: str
    classification_name: str
    feature_name: str
    input_length: int
    input_dtype: DTypeLike = np.float32


class TensorflowModel(ClipClassificationModel):
    callable: Callable
    signature: Signature

    def __init__(
        self,
        callable: Callable,
        signature: Signature,
        tags: List[data.Tag],
        confidence_threshold: float,
        samplerate: int,
        name: str,
        logits: bool = True,
    ):
        self.callable = callable
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.samplerate = samplerate
        self.name = name
        self.signature = signature
        self.input_samples = signature.input_length
        self.num_classes = len(tags)
        self.logits = logits

        _validate_signature(self.callable, self.signature)

    def process_array(self, array: np.ndarray) -> ModelOutput:
        return process_array(
            self.callable,
            self.signature,
            array,
            validate_signature=False,
            logits=self.logits,
        )


def _validate_signature(callable: Callable, signature: Signature) -> None:
    """Validate the signature of a TensorFlow model."""
    function_type = callable.function_type
    parameters = function_type.parameters

    if not len(parameters) == 1:
        raise ValueError("Model must have exactly one input tensor")

    if signature.input_name not in parameters:
        raise ValueError("Input tensor name does not match signature")

    input_param = parameters[signature.input_name]
    input_annotations = input_param.annotation

    if len(input_annotations.shape) != 2:
        raise ValueError("Input tensor must have 2 dimensions")

    if input_annotations.shape[1] != signature.input_length:
        raise ValueError("Input tensor length does not match signature")

    if input_annotations.dtype != signature.input_dtype:
        raise ValueError("Input tensor dtype does not match signature")


def process_array(
    call: Callable,
    signature: Signature,
    array: np.ndarray,
    validate_signature: bool = False,
    logits: bool = True,
) -> ModelOutput:
    """Process an array with a TensorFlow model."""
    if array.ndim == 1:
        array = array[np.newaxis, :]

    if not array.ndim == 2:
        raise ValueError("Input array must have 2 dimensions")

    if not array.shape[1] == signature.input_length:
        raise ValueError(
            "Input array should consist of {} samples".format(
                signature.input_length
            )
        )

    if validate_signature:
        _validate_signature(call, signature)

    if array.dtype != signature.input_dtype:
        array = array.astype(signature.input_dtype)

    output = call(**{signature.input_name: array})

    class_probs = output[signature.classification_name].numpy()  # type: ignore
    features = output[signature.feature_name].numpy()  # type: ignore

    if logits:
        class_probs = flat_sigmoid(class_probs)

    return ModelOutput(class_probs=class_probs, features=features)
