import datetime
from typing import List, Optional

import numpy as np
import xarray as xr
from soundevent import data

from birdnetsnd.constants import DEFAULT_THRESHOLD, HOP_SIZE

__all__ = [
    "convert_to_dataset",
    "convert_to_features_array",
    "convert_to_features_list",
    "convert_to_predicted_tags_list",
    "convert_to_probabilities_array",
]


def convert_to_features_list(
    features: np.ndarray,
    prefix: str = "birdnet_",
) -> List[List[data.Feature]]:
    return [
        [
            data.Feature(name=f"{prefix}{i}", value=feat)
            for i, feat in enumerate(feats)
        ]
        for feats in features
    ]


def convert_to_predicted_tags_list(
    class_probs: np.ndarray,
    tags: List[data.Tag],
    confidence_threshold: float = DEFAULT_THRESHOLD,
) -> List[List[data.PredictedTag]]:
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
    start_time: float = 0,
    hop_size: float = HOP_SIZE,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    recorded_on: Optional[datetime.datetime] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
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
    start_time: float = 0,
    hop_size: float = HOP_SIZE,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    recorded_on: Optional[datetime.datetime] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
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
    start_time: float = 0,
    hop_size: float = HOP_SIZE,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    recorded_on: Optional[datetime.datetime] = None,
    attrs: Optional[dict] = None,
) -> xr.Dataset:
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
