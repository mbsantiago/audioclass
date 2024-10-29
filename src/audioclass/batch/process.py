"""Module for batch processing audio data with classification models."""

from typing import Callable, List, Tuple

import numpy as np
from soundevent import data
from tqdm import tqdm

from audioclass.batch.base import BaseIterator
from audioclass.constants import (
    DEFAULT_THRESHOLD,
)

__all__ = [
    "process_iterable",
]


def process_iterable(
    process_array: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    iterable: BaseIterator,
    tags: List[data.Tag],
    name: str,
    confidence_threshold: float = DEFAULT_THRESHOLD,
) -> List[data.ClipPrediction]:
    """Process an iterable of audio data using a models `process_array` method.

    This function iterates over batches of audio clips, processes each clip
    using the provided `process_array` function, and returns a list of
    `ClipPrediction` objects. The `process_array` function should take a
    numpy array of audio data and return a tuple of class probabilities and
    extracted features.

    Parameters
    ----------
    process_array
        A function that takes a numpy array of audio data and returns a tuple
        of class probabilities and extracted features.
    iterable
        An iterator that yields batches of audio clips.
    tags
        A list of tags that the model can predict.
    name
        The name of the model.
    confidence_threshold
        The minimum confidence threshold for a tag to be included in a
        prediction. Defaults to `DEFAULT_THRESHOLD`.

    Returns
    -------
    List[data.ClipPrediction]
        A list of `ClipPrediction` objects, one for each audio clip.
    """
    hop_size = iterable.input_samples / iterable.samplerate

    output = []
    for array, recordings, frames in tqdm(iterable):
        class_probs, features = process_array(array)

        for probs, feats, recording, frm in zip(
            class_probs,
            features,
            recordings,
            frames.flatten(),
        ):
            clip = data.Clip(
                recording=recording,
                start_time=frm * hop_size,
                end_time=(frm + 1) * hop_size,
            )
            output.append(
                data.ClipPrediction(
                    clip=clip,
                    features=[
                        data.Feature(
                            term=data.term_from_key(f"{name}_{i}"),
                            value=feat,
                        )
                        for i, feat in enumerate(feats)
                    ],
                    tags=[
                        data.PredictedTag(tag=tag, score=score)
                        for tag, score in zip(tags, probs)
                        if score > confidence_threshold
                    ],
                )
            )

    return output
