"""Module for preprocessing audio data.

This module helps loading audio data and preprocessing it into a standardized
format for audio classification models.

Provides functions for loading audio, resampling, and framing into fixed-length
buffers.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from soundevent import arrays, audio, data

__all__ = [
    "load_recording",
    "load_clip",
]


def load_recording(
    recording: data.Recording,
    samplerate: int,
    buffer_size: int,
    audio_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load an audio recording from a soundevent `Recording` object.

    This function will load the audio file, preprocess it, and return a numpy
    array.

    Parameters
    ----------
    recording
        The soundevent `Recording` object representing the audio file.
    samplerate
        The desired sample rate to resample the audio to.
    buffer_size
        The length of each audio frame in samples.
    audio_dir
        The directory containing the audio files. If not provided, the
        recording's default audio directory is used.

    Returns
    -------
    np.ndarray
        A numpy array of shape (num_frames, buffer_size) containing the
        preprocessed audio data.
    """
    wave = audio.load_recording(recording, audio_dir=audio_dir)
    return preprocess_audio(
        wave,
        samplerate=samplerate,
        buffer_size=buffer_size,
    )


def load_clip(
    clip: data.Clip,
    samplerate: int,
    buffer_size: int,
    audio_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Load an audio clip from a soundevent `Clip` object.

    This function will load the clip from the audio file, preprocess it, and
    return a numpy array.

    Parameters
    ----------
    clip : data.Clip
        The soundevent `Clip` object representing the audio segment.
    samplerate : int
        The desired sample rate to resample the audio to.
    buffer_size : int
        The length of each audio frame in samples.
    audio_dir : Optional[Path], optional
        The directory containing the audio files. If not provided, the clip's
        default audio directory is used.

    Returns
    -------
    np.ndarray
        A numpy array of shape (num_frames, buffer_size) containing the
        preprocessed audio data.
    """
    wave = audio.load_clip(clip, audio_dir=audio_dir)
    return preprocess_audio(
        wave,
        samplerate=samplerate,
        buffer_size=buffer_size,
    )


def preprocess_audio(
    wave: xr.DataArray,
    samplerate: int,
    buffer_size: int,
) -> np.ndarray:
    """Preprocess a loaded audio waveform.

    This function performs the following preprocessing steps:

    1. Selects the first channel if multiple channels are present.
    2. Resamples the audio to the specified sample rate.
    3. Frames the audio into fixed-length buffers.

    Parameters
    ----------
    wave : xr.DataArray
        The loaded audio waveform.
    samplerate : int
        The desired sample rate to resample the audio to.
    buffer_size : int
        The length of each audio frame in samples.

    Returns
    -------
    np.ndarray
        A numpy array of shape (num_frames, buffer_size) containing the
        preprocessed audio data.
    """
    if "channel" in wave.dims:
        wave = wave.sel(channel=0)

    wave = resample_audio(wave, samplerate=samplerate)
    return stack_array(wave.data, buffer_size=buffer_size)


def resample_audio(
    wave: xr.DataArray,
    samplerate: int,
) -> xr.DataArray:
    """Resample audio to a specific sample rate.

    Parameters
    ----------
    wave : xr.DataArray
        The audio waveform to resample.
    samplerate : int
        The target sample rate.

    Returns
    -------
    xr.DataArray
        The resampled audio waveform.
    """
    step = arrays.get_dim_step(wave, "time")
    original_samplerate = int(1 / step)

    if original_samplerate == samplerate:
        return wave

    return audio.resample(wave, samplerate)


def stack_array(
    arr: np.ndarray,
    buffer_size: int,
) -> np.ndarray:
    """Stack a 1D array into a 2D array of fixed-length buffers.

    This function pads the input array with zeros if necessary to ensure that
    the number of elements is divisible by the buffer size.

    Parameters
    ----------
    arr : np.ndarray
        The 1D array to stack.
    buffer_size : int
        The length of each buffer.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_buffers, buffer_size) containing the stacked
        buffers.

    Raises
    ------
    ValueError
        If the input array has more than one dimension.
    """
    if arr.ndim != 1:
        raise ValueError(
            f"Wave must have 1 dimension. Found {arr.ndim} dimensions"
        )

    num_samples = arr.size
    num_batches = int(np.ceil(num_samples / buffer_size))
    padded_size = num_batches * buffer_size

    if padded_size > num_samples:
        arr = np.pad(arr, (0, padded_size - num_samples))

    return arr.reshape(num_batches, buffer_size)
