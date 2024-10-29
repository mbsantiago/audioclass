"""Module for defining TensorFlow Lite-based audio classification models.

This module provides classes and functions for creating and using TensorFlow
Lite models for audio classification tasks. It includes a `TFLiteModel` class
that wraps a TensorFlow Lite interpreter and a `Signature` dataclass to define
the model's input and output specifications.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import DTypeLike
from soundevent import data

from audioclass.models.base import ClipClassificationModel, ModelOutput
from audioclass.utils import flat_sigmoid

try:
    from tensorflow._api.v2.lite import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter  # type: ignore

__all__ = [
    "load_model",
    "Signature",
    "TFLiteModel",
    "Interpreter",
]


@dataclass
class Signature:
    """Defines the input and output signature of a TensorFlow Lite model."""

    input_index: int
    """The index of the input tensor in the model."""

    classification_index: int
    """The index of the tensor containing classification probabilities."""

    feature_index: int
    """The index of the tensor containing extracted features."""

    input_length: int
    """The number of audio samples expected in the input tensor."""

    input_dtype: DTypeLike = np.float32
    """The data type of the input tensor. Defaults to np.float32."""


class TFLiteModel(ClipClassificationModel):
    """A wrapper class for TensorFlow Lite audio classification models.

    This class provides a standardized interface for interacting with
    TensorFlow Lite models, allowing them to be used seamlessly with the
    audioclass library.
    """

    interpreter: Interpreter
    """The TensorFlow Lite interpreter object."""

    signature: Signature
    """The input and output signature of the model."""

    def __init__(
        self,
        interpreter: Interpreter,
        signature: Signature,
        tags: List[data.Tag],
        confidence_threshold: float,
        samplerate: int,
        name: str,
        logits: bool = True,
        batch_size: int = 8,
    ):
        """Initialize a TFLiteModel.

        Parameters
        ----------
        interpreter
            The TensorFlow Lite interpreter object.
        signature
            The input and output signature of the model.
        tags
            The list of tags that the model can predict.
        confidence_threshold
            The minimum confidence threshold for assigning a tag to a clip.
        samplerate
            The sample rate of the audio data expected by the model (in Hz).
        name
            The name of the model.
        logits
            Whether the model outputs logits (True) or probabilities (False).
            Defaults to True.
        batch_size
            The maximum number of frames to process in each batch. Defaults to
            8.
        """
        self.interpreter = interpreter
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.samplerate = samplerate
        self.name = name
        self.signature = signature
        self.input_samples = signature.input_length
        self.num_classes = len(tags)
        self.logits = logits
        self.batch_size = batch_size

        _validate_signature(self.interpreter, self.signature)

    def process_array(self, array: np.ndarray) -> ModelOutput:
        """Process a single audio array and return the model output.

        Parameters
        ----------
        array : np.ndarray
            The audio array to be processed, with shape
            `(num_frames, input_samples)`.

        Returns
        -------
        ModelOutput
            A `ModelOutput` object containing the class probabilities and
            extracted features.

        Note
        ----
        This is a low-level method that requires manual batching of
        the input audio array. If you prefer a higher-level
        interface that handles batching automatically, consider
        using `process_file`, `process_recording`, or `process_clip`
        instead.

        Be aware that passing an array with a large batch size may
        exceed available device memory and cause the process to
        crash.
        """
        return process_array(
            self.interpreter,
            self.signature,
            array,
            validate_signature=False,
            logits=self.logits,
        )


def load_model(
    path: Union[Path, str],
    num_threads: Optional[int] = None,
) -> Interpreter:
    """
    Load a TensorFlow Lite model from a file.

    Parameters
    ----------
    path
        The path to the TensorFlow Lite model file.
    num_threads
        The number of threads to use for inference. If None, the default number
        of threads will be used.

    Returns
    -------
    Interpreter
        The TensorFlow Lite interpreter object.
    """
    interpreter = Interpreter(
        model_path=str(path),
        num_threads=num_threads,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    return interpreter


def _validate_signature(
    interpreter: Interpreter, signature: Signature
) -> None:
    """Validate the signature of a TF Lite model.

    Parameters
    ----------
    interpreter
        The TF Lite model interpreter.
    signature
        The input and output signature of the model.

    Raises
    ------
    ValueError
        If the model signature does not match the expected format.
    """
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
    """Process an array with a TF Lite model.

    Parameters
    ----------
    interpreter
        The TF Lite model interpreter.
    signature
        The input and output signature of the model.
    array
        The audio array to be processed, with shape (num_frames, input_samples)
        or (input_samples,).
    validate_signature
        Whether to validate the model signature. Defaults to False.
    logits
        Whether the model outputs logits (True) or probabilities (False).
        Defaults to True.

    Returns
    -------
    ModelOutput
        A `ModelOutput` object containing the class probabilities and extracted
        features.

    Raises
    ------
    ValueError
        If the input array has the wrong shape or if the model signature is
        invalid.
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
