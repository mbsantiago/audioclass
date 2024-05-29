from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import DTypeLike
from soundevent import data
from tflite_runtime.interpreter import Interpreter

from audioclass.models.base import ClipClassificationModel, ModelOutput
from audioclass.utils import flat_sigmoid

__all__ = [
    "load_model",
    "Signature",
    "TFLiteModel",
]


@dataclass
class Signature:
    input_index: int
    classification_index: int
    feature_index: int
    input_length: int
    input_dtype: DTypeLike = np.float32


class TFLiteModel(ClipClassificationModel):
    interpreter: Interpreter
    signature: Signature

    def __init__(
        self,
        interpreter: Interpreter,
        signature: Signature,
        tags: List[data.Tag],
        confidence_threshold: float,
        samplerate: int,
        name: str,
        logits: bool = True,
    ):
        self.interpreter = interpreter
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.samplerate = samplerate
        self.name = name
        self.signature = signature
        self.input_samples = signature.input_length
        self.num_classes = len(tags)
        self.logits = logits

        _validate_signature(self.interpreter, self.signature)

    def process_array(self, array: np.ndarray) -> ModelOutput:
        return process_array(
            self.interpreter,
            self.signature,
            array,
            validate_signature=False,
            logits=self.logits,
        )


def load_model(
    path: Path,
    num_threads: Optional[int] = None,
) -> Interpreter:
    interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def _validate_signature(
    interpreter: Interpreter, signature: Signature
) -> None:
    """Validate the signature of a TF Lite model."""
    input_details = interpreter.get_input_details()

    if not len(input_details) == 1:
        raise ValueError("Model must have exactly one input tensor")

    input_details = input_details[0]

    if not input_details["index"] == signature.input_index:
        raise ValueError("Input tensor index does not match signature")

    if not len(input_details["shape"]) == 2:
        raise ValueError("Input tensor must have 2 dimensions")

    if not input_details["shape"][1] == signature.input_length:
        raise ValueError("Input tensor length does not match signature")

    if not input_details["dtype"] == signature.input_dtype:
        raise ValueError("Input tensor dtype does not match signature")


def process_array(
    interpreter: Interpreter,
    signature: Signature,
    array: np.ndarray,
    validate_signature: bool = False,
    logits: bool = True,
) -> ModelOutput:
    """Process a 2D array with a TF Lite model.

    Parameters
    ----------
    interpreter
        The TF Lite model interpreter.
    array
        An array of audio data with shape (n_samples) or (batch_size, n_samples).
        The number of samples should match the input shape of the model which
        by default is 144000.

    Returns
    -------
    class_probs
        The probability of each class for each sample. This is stored
        in a 2D array with shape (batch_size, n_classes). The values are
        between 0 and 1. The value at index (i, j) is the probability that
        the class j is present in the sample i. The probability scores
        should be interpreted as the confidence of the model in the presence
        of the class in the sample. Caution should be taken when interpreting
        these values as probabilities.
    features
        The features extracted from the audio data. This is stored in a 2D
        array with shape (batch_size, n_features). The features are extracted
        from the last layer of the model and can be used for further analysis.
        The default number of features is 1024.

    Raises
    ------
    ValueError
        If the input array does not have 2 dimensions or if the number of
        samples does not match the input shape of the model.
    """
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
        _validate_signature(interpreter, signature)

    if array.dtype != signature.input_dtype:
        array = array.astype(signature.input_dtype)

    _adjust_dynamic_batch_size(interpreter, array)

    interpreter.set_tensor(signature.input_index, array)
    interpreter.invoke()
    class_probs = interpreter.get_tensor(signature.classification_index)
    features = interpreter.get_tensor(signature.feature_index)

    if logits:
        class_probs = flat_sigmoid(class_probs)

    return ModelOutput(class_probs=class_probs, features=features)


def _adjust_dynamic_batch_size(interpreter, array):
    input_details = interpreter.get_input_details()[0]
    if array.shape[0] != input_details["shape"][0]:
        interpreter.resize_tensor_input(
            input_details["index"],
            array.shape,
        )
        interpreter.allocate_tensors()
