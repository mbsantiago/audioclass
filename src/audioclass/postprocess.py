"""Module for postprocessing audio classification model outputs.

This module provides functions to convert raw model outputs (class
probabilities and features) into various formats suitable for analysis,
visualization, or storage. The primary formats include:

- **xarray Datasets:** Structured datasets containing features, probabilities,
and metadata like time, labels, location, and recording time.
- **Lists of soundevent objects:** Collections of `PredictedTag` and `Feature`
objects, compatible with the soundevent library.

These functions facilitate seamless integration with downstream analysis tools
and enable flexible representation of audio classification results.
"""

import datetime
from typing import List, Optional

import numpy as np
import xarray as xr
from soundevent import data

from audioclass.constants import DEFAULT_THRESHOLD

__all__ = [
    "convert_to_dataset",
    "convert_to_features_array",
    "convert_to_features_list",
    "convert_to_predicted_tags_list",
    "convert_to_probabilities_array",
]


def convert_to_features_list(
    features: np.ndarray,
    prefix: str,
) -> List[List[data.Feature]]:
    """Convert a feature array to a list of soundevent `Feature` objects.

    Parameters
    ----------
    features
        A 2D array of features, where each row corresponds to a frame and each
        column to a feature.
    prefix
        A prefix to add to each feature name.

    Returns
    -------
    List[List[data.Feature]]
        A list of lists of `Feature` objects, where each inner list corresponds
        to a frame and contains the features for that frame.
    """
    return [
        [
            data.Feature(term=data.term_from_key(f"{prefix}{i}"), value=feat)
            for i, feat in enumerate(feats)
        ]
        for feats in features
    ]


def convert_to_predicted_tags_list(
    class_probs: np.ndarray,
    tags: List[data.Tag],
    confidence_threshold: float = DEFAULT_THRESHOLD,
) -> List[List[data.PredictedTag]]:
    """Convert class probabilities to a list of predicted tags.

    Parameters
    ----------
    class_probs
        A 2D array of class probabilities, where each row corresponds to a
        frame and each column to a class.
    tags
        A list of `Tag` objects representing the possible classes.
    confidence_threshold
        The minimum probability threshold for a tag to be considered a
        prediction. Defaults to `DEFAULT_THRESHOLD`.

    Returns
    -------
    List[List[data.PredictedTag]]
        A list of lists of `PredictedTag` objects, where each inner list
        corresponds to a frame and contains the predicted tags for that frame.

    Raises
    ------
    ValueError
        If the number of output tags does not match the number of columns in
        `class_probs`.
    """
    if not class_probs.shape[1] == len(tags):
        raise ValueError(
            "Number of output tags does not match model output shape"
            " ({} != {})".format(len(tags), class_probs.shape[1])
        )
    return [
        [
            data.PredictedTag(tag=tag, score=score)
            for tag, score in zip(tags, probs)
            if score > confidence_threshold
        ]
        for probs in class_probs
    ]


def convert_to_probabilities_array(
    class_probs: np.ndarray,
    labels: List[str],
    hop_size: float,
    start_time: float = 0,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    recorded_on: Optional[datetime.datetime] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    """Convert class probabilities to a DataArray.

    Parameters
    ----------
    class_probs
        A 2D array of class probabilities, where each row corresponds to a
        frame and each column to a class.
    labels
        A list of labels for the classes.
    hop_size
        The time step between frames in seconds.
    start_time
        The start time of the first frame in seconds. Defaults to 0.
    latitude
        The latitude of the recording location. Defaults to None.
    longitude
        The longitude of the recording location. Defaults to None.
    recorded_on
        The date and time the recording was made. Defaults to None.
    attrs
        Additional attributes to add to the DataArray. Defaults to None.

    Returns
    -------
    xr.DataArray
        An xarray DataArray with dimensions `time` and `label`, containing the
        class probabilities.
    """
    times = start_time + np.arange(
        0,
        class_probs.shape[0] * hop_size,
        hop_size,
    )
    return xr.DataArray(
        class_probs,
        coords={
            "time": xr.Variable(
                dims="time",
                data=times,
                attrs={
                    "units": "s",
                    "standard_name": "time",
                    "long_name": "Time from start of recording",
                },
            ),
            "label": labels,
            **_prepare_additional_coords(latitude, longitude, recorded_on),
        },
        dims=["time", "label"],
        attrs=attrs,
    )


def convert_to_features_array(
    features: np.ndarray,
    hop_size: float,
    start_time: float = 0,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    recorded_on: Optional[datetime.datetime] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    """Convert features to an xarray DataArray.

    Parameters
    ----------
    features
        A 2D array of features, where each row corresponds to a frame and each
        column to a feature.
    hop_size
        The time step between frames in seconds.
    start_time
        The start time of the first frame in seconds. Defaults to 0.
    latitude
        The latitude of the recording location. Defaults to None.
    longitude
        The longitude of the recording location. Defaults to None.
    recorded_on
        The date and time the recording was made. Defaults to None.
    attrs
        Additional attributes to add to the DataArray. Defaults to None.

    Returns
    -------
    xr.DataArray
        An xarray DataArray with dimensions `time` and `feature`, containing
        the features.
    """
    times = start_time + np.arange(0, features.shape[0] * hop_size, hop_size)

    return xr.DataArray(
        features,
        coords={
            "time": xr.Variable(
                dims="time",
                data=times,
                attrs={
                    "units": "s",
                    "standard_name": "time",
                    "long_name": "Time from start of recording",
                },
            ),
            "feature": range(features.shape[1]),
            **_prepare_additional_coords(latitude, longitude, recorded_on),
        },
        dims=["time", "feature"],
        attrs=attrs,
    )


def convert_to_dataset(
    features: np.ndarray,
    class_probs: np.ndarray,
    labels: List[str],
    hop_size: float,
    start_time: float = 0,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    recorded_on: Optional[datetime.datetime] = None,
    attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Convert features and class probabilities to an xarray Dataset.

    Parameters
    ----------
    features
        A 2D array of features, where each row corresponds to a frame and each
        column to a feature.
    class_probs
        A 2D array of class probabilities, where each row corresponds to a
        frame and each column to a class.
    labels
        A list of labels for the classes.
    hop_size
        The time step between frames in seconds.
    start_time
        The start time of the first frame in seconds. Defaults to 0.
    latitude
        The latitude of the recording location. Defaults to None.
    longitude
        The longitude of the recording location. Defaults to None.
    recorded_on
        The date and time the recording was made. Defaults to None.
    attrs
        Additional attributes to add to the Dataset. Defaults to None.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the features and probabilities as
        DataArrays, along with coordinates and attributes.
    """
    features_array = convert_to_features_array(
        features,
        start_time=start_time,
        hop_size=hop_size,
        latitude=latitude,
        longitude=longitude,
        recorded_on=recorded_on,
    )

    probabilities_array = convert_to_probabilities_array(
        class_probs,
        labels,
        start_time=start_time,
        hop_size=hop_size,
        latitude=latitude,
        longitude=longitude,
        recorded_on=recorded_on,
    )

    return xr.Dataset(
        {
            "features": features_array,
            "probabilities": probabilities_array,
        },
        attrs=attrs,
    )


def _prepare_additional_coords(
    latitude: Optional[float],
    longitude: Optional[float],
    recorded_on: Optional[datetime.datetime],
) -> dict:
    additional_coords = {}

    if recorded_on is not None:
        additional_coords["recorded_on"] = xr.Variable(
            dims=[],
            data=recorded_on,
            attrs={
                "units": "seconds since 1970-01-01T00:00:00",
                "standard_name": "recorded_on",
                "long_name": "Time of start of recording",
            },
        )

    if latitude is not None:
        additional_coords["latitude"] = xr.Variable(
            dims=[],
            data=latitude,
            attrs={
                "units": "degrees_north",
                "standard_name": "latitude",
                "long_name": "Latitude of recording location",
            },
        )

    if longitude is not None:
        additional_coords["longitude"] = xr.Variable(
            dims=[],
            data=longitude,
            attrs={
                "units": "degrees_east",
                "standard_name": "longitude",
                "long_name": "Longitude of recording location",
            },
        )

    return additional_coords
