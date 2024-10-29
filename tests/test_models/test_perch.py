import datetime

import pytest

pytest.importorskip("tensorflow")

import numpy as np
import xarray as xr
from soundevent import data

from audioclass.models.perch import Perch


@pytest.fixture(scope="module")
def perch() -> Perch:
    return Perch.load()


@pytest.mark.tensorflow
def test_loaded_perch_model_has_correct_signature(perch: Perch):
    assert perch.callable is not None
    assert perch.signature is not None
    assert perch.signature.input_length == 160_000
    assert perch.signature.input_dtype == np.float32
    assert perch.signature.input_name == "inputs"
    assert perch.signature.classification_name == "output_0"
    assert perch.signature.feature_name == "output_1"


@pytest.mark.tensorflow
def test_perch_model_can_process_random_array(perch: Perch):
    array = np.random.rand(1, 160000).astype(np.float32)
    results = perch.process_array(array)
    class_probs, features = results

    assert isinstance(features, np.ndarray)
    assert isinstance(class_probs, np.ndarray)
    assert features.shape == (1, 1280)
    assert class_probs.shape == (1, 10932)


@pytest.mark.tensorflow
def test_perch_model_can_process_file(random_wav_factory, perch: Perch):
    path = random_wav_factory(duration=10)
    clip_predictions = perch.process_file(path)
    assert len(clip_predictions) == 2
    assert all(isinstance(cp, data.ClipPrediction) for cp in clip_predictions)


@pytest.mark.tensorflow
def test_perch_model_can_process_recording(random_wav_factory, perch: Perch):
    path = random_wav_factory(duration=10)
    recording = data.Recording.from_file(path)
    clip_predictions = perch.process_recording(recording)
    assert len(clip_predictions) == 2
    assert all(isinstance(cp, data.ClipPrediction) for cp in clip_predictions)


@pytest.mark.tensorflow
def test_perch_model_can_process_clip(random_wav_factory, perch: Perch):
    path = random_wav_factory(duration=10)
    clip = data.Clip(
        recording=data.Recording.from_file(path),
        start_time=1,
        end_time=6,
    )
    clip_predictions = perch.process_clip(clip)
    assert len(clip_predictions) == 1

    clip_prediction = clip_predictions[0]
    assert isinstance(clip_prediction, data.ClipPrediction)
    assert clip_prediction.clip.start_time == 1
    assert clip_prediction.clip.end_time == 6


@pytest.mark.tensorflow
def test_perch_process_clip_with_dataset_output(
    random_wav_factory, perch: Perch
):
    path = random_wav_factory(duration=10)
    clip = data.Clip(
        recording=data.Recording.from_file(path),
        start_time=1,
        end_time=6,
    )
    dataset = perch.process_clip(clip, fmt="dataset")
    assert isinstance(dataset, xr.Dataset)
    assert "features" in dataset
    assert "probabilities" in dataset


@pytest.mark.tensorflow
def test_perch_can_process_recording_with_metadata(
    random_wav_factory, perch: Perch
):
    path = random_wav_factory(duration=10)
    recording = data.Recording.from_file(
        path,
        latitude=0,
        longitude=0,
        date=datetime.date(2021, 1, 1),
        time=datetime.time(12, 0),
    )
    clip_predictions = perch.process_recording(recording, fmt="dataset")
    assert isinstance(clip_predictions, xr.Dataset)
    assert "features" in clip_predictions
    assert "probabilities" in clip_predictions
    assert "latitude" in clip_predictions
    assert "longitude" in clip_predictions
    assert "recorded_on" in clip_predictions
