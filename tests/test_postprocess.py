import datetime

import numpy as np
import pytest
from soundevent import data

from audioclass.postprocess import (
    convert_to_features_array,
    convert_to_predicted_tags_list,
    convert_to_probabilities_array,
)


def test_fails_to_convert_to_tags_list_if_size_mismatch():
    class_probs = np.array([[0.5, 0.5]])
    tags = [data.Tag(term=data.term_from_key("test"), value="test")]

    with pytest.raises(ValueError, match="(1 != 2)"):
        convert_to_predicted_tags_list(class_probs, tags)


def test_probability_arrays_have_additional_coords():
    class_probs = np.array([[0.5, 0.5]])
    labels = ["test1", "test2"]

    result = convert_to_probabilities_array(
        class_probs,
        labels,
        hop_size=1.0,
        latitude=0.0,
        longitude=0.0,
        recorded_on=datetime.datetime.now(),
    )

    assert "latitude" in result.coords
    assert "longitude" in result.coords
    assert "recorded_on" in result.coords


def test_feature_arrays_have_additional_coords():
    features = np.array([[0.5, 0.5]])
    result = convert_to_features_array(
        features,
        hop_size=1.0,
        latitude=0.0,
        longitude=0.0,
        recorded_on=datetime.datetime.now(),
    )

    assert "latitude" in result.coords
    assert "longitude" in result.coords
    assert "recorded_on" in result.coords
