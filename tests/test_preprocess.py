import numpy as np
import pytest
from soundevent import audio, data

from audioclass.preprocess import (
    load_clip,
    load_recording,
    resample_audio,
    stack_array,
)


def test_stack_array_fails_if_not_one_dimensional():
    array = np.ones([1, 2])

    with pytest.raises(ValueError):
        stack_array(array, buffer_size=2)


def test_stack_array_expands_with_zeros():
    array = np.array([1, 2, 3, 4, 5])
    buffer_size = 10
    stacked = stack_array(array, buffer_size=buffer_size)
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (1, buffer_size)
    assert np.all(stacked[0, :5] == array)
    assert np.all(stacked[0, 5:] == 0)


def test_stack_array_creates_multiple_stacks_if_too_long():
    array = np.arange(20)
    buffer_size = 10
    stacked = stack_array(array, buffer_size=buffer_size)
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (2, buffer_size)
    assert np.all(stacked[0, :] == array[:buffer_size])
    assert np.all(stacked[1, :] == array[buffer_size:])


@pytest.mark.parametrize("samplerate", [16_000, 48_000, 44_100])
def test_preprocess_does_not_resample_if_unnecessary(
    samplerate: int, random_wav_factory, mocker
):
    resample_fn = mocker.patch("soundevent.audio.resample")

    path = random_wav_factory(duration=0.5, samplerate=samplerate)
    recording = data.Recording.from_file(path)
    wave = audio.load_recording(recording)
    resample_audio(wave, samplerate=samplerate)

    resample_fn.assert_not_called()


def test_preprocess_resamples_if_necessary(random_wav_factory, mocker):
    resample_fn = mocker.patch("soundevent.audio.resample")

    path = random_wav_factory(duration=0.5, samplerate=16_000)
    recording = data.Recording.from_file(path)
    wave = audio.load_recording(recording)

    resample_audio(wave, samplerate=48_000)

    resample_fn.assert_called_once_with(wave, 48_000)


def test_can_process_multi_channel_recording(random_wav_factory):
    path = random_wav_factory(duration=0.5, channels=2)
    recording = data.Recording.from_file(path)
    arr = load_recording(
        recording,
        samplerate=48_000,
        buffer_size=144_000,
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 144_000)


def test_can_load_audio_from_file(random_wav_factory):
    path = random_wav_factory(duration=0.5)
    arr = load_recording(
        data.Recording.from_file(path),
        samplerate=48_000,
        buffer_size=144_000,
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 144_000)


def test_can_load_clip(random_wav_factory):
    path = random_wav_factory(duration=10)
    clip = data.Clip(
        recording=data.Recording.from_file(path),
        start_time=1,
        end_time=4,
    )
    arr = load_clip(clip, samplerate=48_000, buffer_size=144_000)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 144_000)
