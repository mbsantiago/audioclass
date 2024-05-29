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
                        data.Feature(name=f"{name}_{i}", value=feat)
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
