"""Module for defining TensorFlow-based audio classification models.

This module provides classes and functions for creating and using TensorFlow
models for audio classification tasks. It includes a `TensorflowModel` class
that wraps a TensorFlow callable and a `Signature` dataclass to define the
model's input and output specifications.
"""

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
    """Defines the input and output signature of a TensorFlow model."""

    input_name: str
    """The name of the input tensor."""

    classification_name: str
    """The name of the output tensor containing classification probabilities."""

    feature_name: str
    """The name of the output tensor containing extracted features."""

    input_length: int
    """The number of samples expected in the input tensor."""

    input_dtype: DTypeLike = np.float32
    """The data type of the input tensor. Defaults to np.float32."""


class TensorflowModel(ClipClassificationModel):
    """A wrapper class for TensorFlow audio classification models.

    This class provides a standardized interface for interacting with
    TensorFlow models, allowing them to be used seamlessly with the audioclass
    library.
    """

    callable: Callable
    """The TensorFlow callable representing the model."""

    signature: Signature
    """The input and output signature of the model."""

    def __init__(
        self,
        callable: Callable,
        signature: Signature,
        tags: List[data.Tag],
        confidence_threshold: float,
        samplerate: int,
        name: str,
        logits: bool = True,
        batch_size: int = 8,
    ):
        """Initialize a TensorflowModel.

        Parameters
        ----------
        callable
            The TensorFlow callable representing the model.
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
            The maximum number of frames to process in each batch.
            Defaults to 8.
        """
        self.callable = callable
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.samplerate = samplerate
        self.name = name
        self.signature = signature
        self.input_samples = signature.input_length
        self.num_classes = len(tags)
        self.logits = logits
        self.batch_size = batch_size

        _validate_signature(self.callable, self.signature)

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
            self.callable,
            self.signature,
            array,
            validate_signature=False,
            logits=self.logits,
        )


def _validate_signature(callable: Callable, signature: Signature) -> None:
    """Validate the signature of a TensorFlow model.

    Parameters
    ----------
    callable : Callable
        The TensorFlow callable representing the model.
    signature : Signature
        The input and output signature of the model.

    Raises
    ------
    ValueError
        If the model signature does not match the expected format.
    """
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
    """Process an array with a TensorFlow model.

    Parameters
    ----------
    call
        The TensorFlow callable representing the model.
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
        _validate_signature(call, signature)

    if array.dtype != signature.input_dtype:
        array = array.astype(signature.input_dtype)

    output = call(**{signature.input_name: array})

    class_probs = output[signature.classification_name].numpy()  # type: ignore
    features = output[signature.feature_name].numpy()  # type: ignore

    if logits:
        class_probs = flat_sigmoid(class_probs)

    return ModelOutput(class_probs=class_probs, features=features)
