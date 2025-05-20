"""Audio Classification Module for Bird Sound Detection.

This module provides tools for loading and using the BirdNET sound event
detection model. BirdNET is a deep learning model designed to classify bird
species from audio recordings.
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import tensorflow as tf
from soundevent import data, terms

from audioclass.constants import DATA_DIR, DEFAULT_THRESHOLD
from audioclass.models.birdnet import LABELS_PATH
from audioclass.models.tensorflow import Signature, TensorflowModel
from audioclass.utils import load_artifact

INPUT_SAMPLES = 144000
"""Default number of samples expected in the input tensor.

This value corresponds to 3 seconds of audio data at a sample rate of 48,000
Hz.
"""

SAMPLERATE = 48_000
"""Default sample rate of the audio data expected by the model (in Hz).

This value corresponds to the sample rate used by the BirdNET model.
"""

SAVED_MODEL_PATH = DATA_DIR / "BirdNET_GLOBAL_6K_V2.4"


class BirdNETAnalyzer(TensorflowModel):
    """BirdNET sound event detection model.

    This class loads and wraps the BirdNET TensorFlow SavedModel for bird
    sound classification and embedding extraction.
    """

    @classmethod
    def load(
        cls,
        model_url: Union[Path, str] = SAVED_MODEL_PATH,
        tags_url: Union[Path, str] = LABELS_PATH,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        samplerate: int = SAMPLERATE,
        name: str = "BirdNET",
        common_name: bool = False,
        batch_size: int = 8,
    ):
        """Load a BirdNET model from a saved model directory.

        Parameters
        ----------
        model_url
            The path to the saved model directory. Defaults to a local copy of the BirdNET model.
        tags_url
            The URL or path to the file containing the labels. Defaults to the labels file in the BirdNET repository.
        confidence_threshold
            The minimum confidence threshold for making predictions. Defaults to `DEFAULT_THRESHOLD`.
        samplerate
            The sample rate of the audio data expected by the model (in Hz). Defaults to `SAMPLERATE`.
        name
            The name of the model. Defaults to "BirdNET".

        Returns
        -------
        BirdNETAnalyzer
            An instance of the BirdNET model.
        """
        tags = load_tags(tags_url, common_name=common_name)
        model = tf.saved_model.load(model_url)

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[None, 144000],  # type: ignore
                    dtype=tf.float32,  # type: ignore
                    name="inputs",  # type: ignore
                )
            ]
        )
        def callable(inputs):
            classification = model.basic(inputs)["scores"]  # type: ignore
            embeddings = model.embeddings(inputs)["embeddings"]  # type: ignore
            return dict(classification=classification, embeddings=embeddings)

        return cls(
            callable,  # type: ignore
            signature=Signature(
                input_name="inputs",
                classification_name="classification",
                feature_name="embeddings",
                input_length=INPUT_SAMPLES,
                input_dtype=np.float32,
            ),
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
