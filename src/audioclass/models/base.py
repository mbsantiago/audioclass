"""Module defining the base classes for audio classification models and their output format.

This module provides abstract classes for clip classification models,
establishing a standard interface for model input, processing, and output. It
also defines the structure of the model output, which includes class
probabilities and extracted features.
"""

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, NamedTuple, Union, overload

import numpy as np
import xarray as xr
from soundevent import data

from audioclass.batch import BaseIterator, process_iterable
from audioclass.postprocess import (
    convert_to_dataset,
    convert_to_features_list,
    convert_to_predicted_tags_list,
)
from audioclass.preprocess import load_clip
from audioclass.utils import batched

__all__ = [
    "ModelOutput",
    "ClipClassificationModel",
]


class ModelOutput(NamedTuple):
    """Output format for audio classification models."""

    class_probs: np.ndarray
    """Array of class probabilities for each frame.

    The array has shape `(num_frames, num_classes)`, where `num_frames` is the
    number of frames in the input audio clip and `num_classes` is the number of
    classes that the model can predict.

    Notice that the interpretation may vary depending on the model and it is
    advisable to check the model's documentation for more information.
    """

    features: np.ndarray
    """Array of extracted features for each frame.

    The array has shape `(num_frames, num_features)`, where `num_frames` is the
    number of frames in the input audio clip and `num_features` is the number
    of features extracted by the model.

    The features can be used for further analysis or visualization of the model
    output.
    """


class ClipClassificationModel(ABC):
    """Abstract base class for audio clip classification models.

    This class defines the common interface for audio classification models
    that process individual clips. It provides methods for processing raw audio
    arrays, files, recordings, and clips, as well as an iterable of clips.
    """

    name: str
    """The name of the model."""

    samplerate: int
    """The sample rate of the audio data expected by the model (in Hz)."""

    input_samples: int
    """The number of audio samples expected in each input frame."""

    num_classes: int
    """The number of classes that the model can predict."""

    confidence_threshold: float
    """The minimum confidence threshold for a class to be considered."""

    tags: List[data.Tag]
    """The list of tags that the model can predict."""

    batch_size: int = 8
    """The maximum number of framces to process in each batch."""

    @abstractmethod
    def process_array(self, array: np.ndarray) -> ModelOutput:
        """Process a single audio array and return the model output.

        Parameters
        ----------
        array
            The audio array to be processed, with shape
            `(num_frames, input_samples)`.

        Returns
        -------
        ModelOutput
            A `ModelOutput` object containing the class probabilities and
            extracted features.
        """

    @overload
    def process_file(  # type: ignore # pragma: no cover
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
        """Process an audio file and return the model output.

        Parameters
        ----------
        path
            The path to the audio file.
        fmt
            The desired output format. "soundevent" returns a list of
            `ClipPrediction` objects, while "dataset" returns an xarray
            `Dataset`. Defaults to "soundevent".
        **kwargs
            Additional keyword arguments to pass to `Recording.from_file()`.

        Returns
        -------
        Union[List[data.ClipPrediction], xr.Dataset]
            The model output in the specified format.
        """
        recording = data.Recording.from_file(path, **kwargs)
        return self.process_recording(recording, fmt=fmt)

    @overload
    def process_recording(  # type: ignore # pragma: no cover
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
        """Process an audio recording and return the model output.

        Parameters
        ----------
        recording
            The `Recording` object representing the audio.
        fmt
            The desired output format. "soundevent" returns a list of
            `ClipPrediction` objects, while "dataset" returns an xarray
            `Dataset`. Defaults to "soundevent".

        Returns
        -------
        Union[List[data.ClipPrediction], xr.Dataset]
            The model output in the specified format.
        """
        clip = data.Clip(
            recording=recording,
            start_time=0,
            end_time=recording.duration,
        )
        return self.process_clip(clip, fmt=fmt)

    @overload
    def process_clip(  # type: ignore # pragma: no cover
        self,
        clip: data.Clip,
        fmt: Literal["soundevent"] = "soundevent",
    ) -> List[data.ClipPrediction]: ...

    @overload
    def process_clip(  # pragma: no cover
        self,
        clip: data.Clip,
        fmt: Literal["dataset"] = "dataset",
    ) -> xr.Dataset: ...

    def process_clip(
        self,
        clip: data.Clip,
        fmt: Literal["soundevent", "dataset"] = "soundevent",
    ) -> Union[List[data.ClipPrediction], xr.Dataset]:
        """Process an audio clip and return the model output.

        Parameters
        ----------
        clip
            The `Clip` object representing the audio segment.
        fmt
            The desired output format. "soundevent" returns a list of
            `ClipPrediction` objects, while "dataset" returns an xarray
            `Dataset`. Defaults to "soundevent".

        Returns
        -------
        Union[List[data.ClipPrediction], xr.Dataset]
            The model output in the specified format.
        """
        array = load_clip(
            clip,
            samplerate=self.samplerate,
            buffer_size=self.input_samples,
        )

        class_probs = []
        features = []
        for batch in batched(array, self.batch_size):
            probs, feats = self.process_array(batch)
            class_probs.append(probs)
            features.append(feats)

        class_probs = np.concatenate(class_probs)
        features = np.concatenate(features)

        if fmt == "soundevent":
            return self._convert_to_soundevent(clip, class_probs, features)

        return self._convert_to_dataset(clip, class_probs, features)

    def process_iterable(
        self,
        iterable: BaseIterator,
    ) -> List[data.ClipPrediction]:
        """Process an iterable of audio clips and return a list of predictions.

        Parameters
        ----------
        iterable
            An iterator that yields `Clip` objects.

        Returns
        -------
        List[data.ClipPrediction]
            A list of `ClipPrediction` objects, one for each clip in the
            iterable.
        """
        return process_iterable(
            self.process_array,
            iterable,
            self.tags,
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
