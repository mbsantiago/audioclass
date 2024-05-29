"""Module for loading and using the Google Perch audio classification model.

This module provides a convenient interface for working with the Perch model, a
TensorFlow Hub-based model designed for bird sound classification. It includes
the `Perch` class, which is a subclass of `TensorflowModel`, and functions for
loading the model and its associated labels.

Notes
-----
The Perch model is hosted on Kaggle. Depending on your network configuration,
you might need to set up Kaggle API credentials to access the model. Refer to
Kaggle's documentation for instructions.

This package is not affiliated with Google Research, the original developers of
the Perch model.
"""

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
"""Default sample rate of the audio data expected by the model (in Hz)."""

INPUT_SAMPLES = 160_000
"""Default number of samples expected in the input tensor.

This value corresponds to 5 seconds of audio data at a sample rate of 32,000
Hz.
"""

MODEL_PATH = "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4"
"""Default path to the Perch TensorFlow Hub model URL."""

TAGS_PATH = DATA_DIR / "perch" / "label.csv"
"""Default path to the Perch labels file."""


class Perch(TensorflowModel):
    """Google Perch audio classification model.

    This class is a wrapper around a TensorFlow Hub model for bird sound
    classification. It provides methods for loading the model, processing audio
    data, and returning predictions.
    """

    @classmethod
    def load(
        cls,
        model_url: Union[Path, str] = MODEL_PATH,
        tags_url: Union[Path, str] = TAGS_PATH,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        samplerate: int = SAMPLERATE,
        name: str = "Perch",
    ):
        """Load a Perch model from a URL.

        Parameters
        ----------
        model_url
            The URL of the TensorFlow Hub model. Defaults to the official Perch model URL.
        tags_url
            The URL or path to the file containing the labels. Defaults to the tags file included in the package.
        confidence_threshold
            The minimum confidence threshold for making predictions. Defaults to `DEFAULT_THRESHOLD`.
        samplerate
            The sample rate of the audio data expected by the model (in Hz). Defaults to `SAMPLERATE`.
        name
            The name of the model. Defaults to "Perch".

        Returns
        -------
        Perch
            An instance of the Perch class.
        """
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
    """
    Get the signature of a Perch model.

    Parameters
    ----------
    callable
        The TensorFlow callable representing the model.

    Returns
    -------
    Signature
        The signature of the Perch model.
    """
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
    """Load Perch labels from a file.

    Parameters
    ----------
    path
        Path or URL to the file containing the labels. Defaults to the tags
        file included in the package.

    Returns
    -------
    List[data.Tag]
        List of soundevent `Tag` objects.
    """
    path = load_artifact(path)
    tags = path.read_text().splitlines()[1:]
    return [data.Tag(key="ebird2021", value=tag) for tag in tags]
