import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, overload

import numpy as np
import xarray as xr
from soundevent import data
from tflite_runtime.interpreter import Interpreter

from birdnetsnd.constants import (
    DEFAULT_THRESHOLD,
    LABELS_PATH,
    MODEL_PATH,
    SAMPLERATE,
)
from birdnetsnd.postprocess import (
    convert_to_dataset,
    convert_to_features_list,
    convert_to_predicted_tags_list,
)
from birdnetsnd.preprocess import (
    load_clip,
)
from birdnetsnd.process import process_array

__all__ = [
    "BirdNET",
    "load_model",
    "load_tags",
]


def load_model(
    path: Path = MODEL_PATH,
    num_threads: Optional[int] = None,
) -> Interpreter:
    interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def load_tags(
    path: Path = LABELS_PATH,
    common_name: bool = False,
) -> List[data.Tag]:
    """Load labels from a file.

    Parameters
    ----------
    path : Path
        Path to the file containing the labels.
    common_name : bool, optional
        Whether to return the common name instead of the scientific name, by
        default False

    Returns
    -------
    List[str]
        List of labels.

    Notes
    -----
    The file should contain one label per line and should be in the format:

    ```
    <scientific_name>_<common_name>
    ```
    """
    with open(path) as f:
        labels = f.read().splitlines()

    index = 1 if common_name else 0
    key = "common_name" if common_name else "scientific_name"
    return [
        data.Tag(key=key, value=label.split("_")[index]) for label in labels
    ]


class BirdNET:
    name: str = "BirdNET"
    samplerate: int
    input_samples: int
    num_classes: int
    interpreter: Interpreter
    tags: List[data.Tag]

    def __init__(
        self,
        interpreter: Interpreter,
        tags: List[data.Tag],
        confidence_threshold: float = DEFAULT_THRESHOLD,
        samplerate: int = SAMPLERATE,
    ):
        self.interpreter = interpreter
        self.tags = tags
        self.confidence_threshold = confidence_threshold
        self.samplerate = samplerate

        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        self.num_classes = output_details["shape"][-1]
        self.input_samples = input_details["shape"][-1]

        if not self.num_classes == len(tags):
            raise ValueError(
                "Number of output labels does not match model output shape"
            )

    @classmethod
    def from_model_file(
        cls,
        model_path: Path = MODEL_PATH,
        labels_path: Path = LABELS_PATH,
        num_threads: Optional[int] = None,
        confidence_threshold: float = 0.1,
        common_name: bool = False,
    ) -> "BirdNET":
        interpreter = load_model(model_path, num_threads)
        tags = load_tags(labels_path, common_name=common_name)
        return cls(
            interpreter=interpreter,
            tags=tags,
            confidence_threshold=confidence_threshold,
        )

    def process_array(
        self,
        array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return process_array(self.interpreter, array)

    @overload
    def process_file(  # pragma: no cover
        self,
        path: Path,
        fmt: Literal["soundevent"] = "soundevent",
        **kwargs,
    ) -> List[data.ClipPrediction]: ...

    @overload
    def process_file(  # pragma: no cover
        self,
        path: Path,
        fmt: Literal["dataset"] = "dataset",
        **kwargs,
    ) -> xr.Dataset: ...

    def process_file(
        self,
        path: Path,
        fmt: Literal["soundevent", "dataset"] = "soundevent",
        **kwargs,
    ) -> Union[List[data.ClipPrediction], xr.Dataset]:
        recording = data.Recording.from_file(path, **kwargs)
        return self.process_recording(recording, fmt=fmt)

    @overload
    def process_recording(  # pragma: no cover
        self,
        recording: data.Recording,
        fmt: Literal["soundevent"] = "soundevent",
    ) -> List[data.ClipPrediction]: ...

    @overload
    def process_recording(  # pragma: no cover
        self,
        recording: data.Recording,
        fmt: Literal["dataset"] = "dataset",
    ) -> xr.Dataset: ...

    def process_recording(
        self,
        recording: data.Recording,
        fmt: Literal["soundevent", "dataset"] = "soundevent",
    ) -> Union[List[data.ClipPrediction], xr.Dataset]:
        clip = data.Clip(
            recording=recording,
            start_time=0,
            end_time=recording.duration,
        )
        return self.process_clip(clip, fmt=fmt)

    @overload
    def process_clip(
        self,
        clip: data.Clip,
        fmt: Literal["soundevent"] = "soundevent",
    ) -> List[data.ClipPrediction]: ...

    @overload
    def process_clip(
        self,
        clip: data.Clip,
        fmt: Literal["dataset"] = "dataset",
    ) -> xr.Dataset: ...

    def process_clip(
        self,
        clip: data.Clip,
        fmt: Literal["soundevent", "dataset"] = "soundevent",
    ) -> Union[List[data.ClipPrediction], xr.Dataset]:
        array = load_clip(
            clip,
            samplerate=self.samplerate,
            buffer_size=self.input_samples,
        )
        class_probs, features = self.process_array(array)

        if fmt == "soundevent":
            return self._convert_to_soundevent(clip, class_probs, features)

        return self._convert_to_dataset(clip, class_probs, features)

    def _convert_to_dataset(
        self,
        clip: data.Clip,
        class_probs: np.ndarray,
        features: np.ndarray,
    ) -> xr.Dataset:
        recorded_on = clip.recording.date

        if recorded_on is not None:
            recorded_on = datetime.datetime.combine(
                recorded_on,
                clip.recording.time or datetime.time.min,
            )

        return convert_to_dataset(
            features,
            class_probs,
            labels=[tag.value for tag in self.tags],
            start_time=clip.start_time,
            hop_size=self.input_samples / self.samplerate,
            latitude=clip.recording.latitude,
            longitude=clip.recording.longitude,
            recorded_on=recorded_on,
        )

    def _convert_to_soundevent(
        self,
        clip: data.Clip,
        class_probs: np.ndarray,
        features: np.ndarray,
    ) -> List[data.ClipPrediction]:
        predicted_tags = convert_to_predicted_tags_list(
            class_probs,
            self.tags,
            self.confidence_threshold,
        )

        features_list = convert_to_features_list(
            features,
            prefix=self.name + "_",
        )

        hop_size = self.input_samples / self.samplerate

        return [
            data.ClipPrediction(
                clip=data.Clip(
                    recording=clip.recording,
                    start_time=clip.start_time + index * hop_size,
                    end_time=clip.start_time + (index + 1) * hop_size,
                ),
                tags=tags,
                features=feats,
            )
            for index, (tags, feats) in enumerate(
                zip(predicted_tags, features_list)
            )
        ]
