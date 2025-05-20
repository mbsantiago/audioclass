"""Module for loading and using the BirdNET audio classification model.

This module provides a convenient interface for working with the BirdNET model,
which is a TensorFlow Lite-based model designed for bird sound classification.
It includes the `BirdNET` class, which is a subclass of `TFLiteModel`, and
functions for loading the model and its associated labels.

Notes
-----
The BirdNET model was developed by the K. Lisa Yang Center for Conservation
Bioacoustics at the Cornell Lab of Ornithology, in collaboration with Chemnitz
University of Technology. This package is not affiliated with the BirdNET
project.

BirdNET is licensed under a Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License.

If you use the BirdNET model, please cite:

    Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep
    learning solution for avian diversity monitoring. Ecological Informatics,
    61, 101236.

For further details, please visit the official
[BirdNET repository](https://github.com/kahst/BirdNET-Analyzer)
"""

from pathlib import Path
from typing import List, Optional, Union

from soundevent import data, terms

from audioclass.constants import DEFAULT_THRESHOLD
from audioclass.models.tflite import (
    Interpreter,
    Signature,
    TFLiteModel,
    load_model,
)
from audioclass.utils import load_artifact

__all__ = ["BirdNET"]

INPUT_SAMPLES = 144000
"""Default number of samples expected in the input tensor.

This value corresponds to 3 seconds of audio data at a sample rate of 48,000
Hz.
"""

SAMPLERATE = 48_000
"""Default sample rate of the audio data expected by the model (in Hz).

This value corresponds to the sample rate used by the BirdNET model.
"""

MODEL_PATH: str = "https://github.com/birdnet-team/BirdNET-Analyzer/raw/refs/tags/v1.5.1/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
"""Default path to the BirdNET TensorFlow Lite model file."""

LABELS_PATH = "https://github.com/birdnet-team/BirdNET-Analyzer/raw/refs/tags/v2.0.0/birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt"
"""Default path to the BirdNET labels file."""


class BirdNET(TFLiteModel):
    """BirdNET audio classification model.

    This class is a wrapper around a TensorFlow Lite model for bird sound
    classification. It provides methods for loading the model, processing audio
    data, and returning predictions.
    """

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
        batch_size: int = 8,
    ) -> "BirdNET":
        """Load a BirdNET model from a file or URL.

        Parameters
        ----------
        model_path
            The path or URL to the TensorFlow Lite model file. Defaults to the
            latest version of the BirdNET model.
        labels_path
            The path or URL to the labels file. Defaults to the latest version
            of the BirdNET labels.
        num_threads
            The number of threads to use for inference. If None, the default
            number of threads will be used.
        confidence_threshold
            The minimum confidence threshold for making predictions. Defaults
            to `DEFAULT_THRESHOLD`.
        samplerate
            The sample rate of the audio data expected by the model (in Hz).
            Defaults to 48,000 Hz.
        name
            The name of the model. Defaults to "BirdNET".
        common_name
            Whether to use common names for bird species instead of scientific
            names. Defaults to False.
        batch_size
            The number of samples to process in each batch. Defaults to 8.

        Returns
        -------
        BirdNET
            An instance of the BirdNET class.
        """
        model_path = load_artifact(model_path)
        labels_path = load_artifact(labels_path)
        interpreter = load_model(model_path, num_threads)
        tags = load_tags(labels_path, common_name=common_name)
        return cls(
            interpreter=interpreter,
            signature=get_signature(interpreter),  # type: ignore
            tags=tags,
            confidence_threshold=confidence_threshold,
            samplerate=samplerate,
            name=name,
            batch_size=batch_size,
        )


def load_tags(
    path: Union[Path, str] = LABELS_PATH,
    common_name: bool = False,
) -> List[data.Tag]:
    """
    Load BirdNET labels from a file.

    Parameters
    ----------
    path
        Path or URL to the file containing the labels. Defaults to the latest
        version of the BirdNET labels.
    common_name
        Whether to return the common name instead of the scientific name.
        Defaults to False.

    Returns
    -------
    List[data.Tag]
        List of soundevent `Tag` objects.
    """
    path = load_artifact(path)

    with open(path) as f:
        labels = f.read().splitlines()

    index = 1 if common_name else 0
    term = terms.common_name if common_name else terms.scientific_name
    return [
        data.Tag(term=term, value=label.split("_")[index]) for label in labels
    ]


def get_signature(interpreter: Interpreter) -> Signature:
    """Get the signature of a BirdNET model.

    Parameters
    ----------
    interpreter : Interpreter
        The TensorFlow Lite interpreter object.

    Returns
    -------
    Signature
        The signature of the BirdNET model.

    Raises
    ------
    ValueError
        If the model signature does not match the expected format.
    """
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
