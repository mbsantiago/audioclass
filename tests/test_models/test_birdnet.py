import datetime

import numpy as np
import pytest
import xarray as xr
from soundevent import data

pytest.importorskip("tflite_runtime")

from audioclass.models.birdnet import BirdNET


@pytest.mark.tflite
def test_can_instantiate_birdnet_model_from_model_files():
    model = BirdNET.load()
    assert isinstance(model, BirdNET)


@pytest.mark.tflite
def test_birdnet_model_can_process_random_array():
    model = BirdNET.load(confidence_threshold=0)
    array = np.random.rand(1, 144000).astype(np.float32)
    results = model.process_array(array)
    class_probs, features = results

    assert isinstance(features, np.ndarray)
    assert isinstance(class_probs, np.ndarray)
    assert features.shape == (1, 1024)
    assert class_probs.shape == (1, 6522)


@pytest.mark.tflite
def test_birdnet_model_can_process_file(random_wav_factory):
    model = BirdNET.load()
    path = random_wav_factory(duration=10)
    clip_predictions = model.process_file(path)
    assert len(clip_predictions) == 4
    assert all(isinstance(cp, data.ClipPrediction) for cp in clip_predictions)


@pytest.mark.tflite
def test_birdnet_model_can_process_recording(random_wav_factory):
    model = BirdNET.load()
    path = random_wav_factory(duration=10)
    recording = data.Recording.from_file(path)
    clip_predictions = model.process_recording(recording)
    assert len(clip_predictions) == 4
    assert all(isinstance(cp, data.ClipPrediction) for cp in clip_predictions)


@pytest.mark.tflite
def test_birdnet_model_can_process_clip(random_wav_factory):
    model = BirdNET.load()
    path = random_wav_factory(duration=10)
    clip = data.Clip(
        recording=data.Recording.from_file(path),
        start_time=1,
        end_time=4,
    )
    clip_predictions = model.process_clip(clip)
    assert len(clip_predictions) == 1

    clip_prediction = clip_predictions[0]
    assert isinstance(clip_prediction, data.ClipPrediction)
    assert clip_prediction.clip.start_time == 1
    assert clip_prediction.clip.end_time == 4


@pytest.mark.tflite
def test_birdnet_process_clip_with_dataset_output(random_wav_factory):
    model = BirdNET.load()
    path = random_wav_factory(duration=10)
    clip = data.Clip(
        recording=data.Recording.from_file(path),
        start_time=1,
        end_time=4,
    )
    dataset = model.process_clip(clip, fmt="dataset")
    assert isinstance(dataset, xr.Dataset)
    assert "features" in dataset
    assert "probabilities" in dataset


@pytest.mark.tflite
def test_birdnet_can_process_recording_with_metadata(random_wav_factory):
    model = BirdNET.load()
    path = random_wav_factory(duration=10)
    recording = data.Recording.from_file(
        path,
        latitude=0,
        longitude=0,
        date=datetime.date(2021, 1, 1),
        time=datetime.time(12, 0),
    )
    clip_predictions = model.process_recording(recording, fmt="dataset")
    assert isinstance(clip_predictions, xr.Dataset)
    assert "features" in clip_predictions
    assert "probabilities" in clip_predictions
    assert "latitude" in clip_predictions
    assert "longitude" in clip_predictions
    assert "recorded_on" in clip_predictions


@pytest.mark.tflite
def test_birdnet_can_process_a_long_recording(random_wav_factory):
    model = BirdNET.load()
    path = random_wav_factory(duration=150)
    predictions = model.process_file(path)
    assert isinstance(predictions, list)
