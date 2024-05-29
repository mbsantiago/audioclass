import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, NamedTuple, Union, overload

import numpy as np
import xarray as xr
from soundevent import data

from audioclass.batch import BatchGenerator, process_iterable
from audioclass.postprocess import (
    convert_to_dataset,
    convert_to_features_list,
    convert_to_predicted_tags_list,
)
from audioclass.preprocess import (
    load_clip,
)

__all__ = [
    "ModelOutput",
    "ClipClassificationModel",
]


class ModelOutput(NamedTuple):
    class_probs: np.ndarray
    features: np.ndarray


class ClipClassificationModel(ABC):
    name: str
    samplerate: int
    input_samples: int
    num_classes: int
    confidence_threshold: float
    tags: List[data.Tag]

    @abstractmethod
    def process_array(self, array: np.ndarray) -> ModelOutput: ...

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

    def process_iterable(
        self,
        iterable: BatchGenerator,
    ) -> List[data.ClipPrediction]:
        return process_iterable(
            self.process_array,
            iterable,
            self.tags,
            samplerate=self.samplerate,
            input_samples=self.input_samples,
            name=self.name,
            confidence_threshold=self.confidence_threshold,
        )

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
