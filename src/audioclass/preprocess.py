from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from soundevent import arrays, audio, data

from audioclass.constants import INPUT_SAMPLES, SAMPLERATE

__all__ = [
    "load_recording",
    "load_clip",
]


def load_recording(
    recording: data.Recording,
    samplerate: int = SAMPLERATE,
    buffer_size: int = INPUT_SAMPLES,
    audio_dir: Optional[Path] = None,
) -> np.ndarray:
    wave = audio.load_recording(recording, audio_dir=audio_dir)
    return preprocess_audio(
        wave,
        samplerate=samplerate,
        buffer_size=buffer_size,
    )


def load_clip(
    clip: data.Clip,
    samplerate: int = SAMPLERATE,
    buffer_size: int = INPUT_SAMPLES,
    audio_dir: Optional[Path] = None,
) -> np.ndarray:
    wave = audio.load_clip(clip, audio_dir=audio_dir)
    return preprocess_audio(
        wave,
        samplerate=samplerate,
        buffer_size=buffer_size,
    )


def preprocess_audio(
    wave: xr.DataArray,
    samplerate: int = SAMPLERATE,
    buffer_size: int = INPUT_SAMPLES,
) -> np.ndarray:
    if "channel" in wave.dims:
        wave = wave.sel(channel=0)

    wave = resample_audio(wave, samplerate=samplerate)
    return stack_array(wave.data, buffer_size=buffer_size)


def resample_audio(
    wave: xr.DataArray,
    samplerate: int = SAMPLERATE,
) -> xr.DataArray:
    step = arrays.get_dim_step(wave, "time")
    original_samplerate = int(1 / step)

    if original_samplerate == samplerate:
        return wave

    return audio.resample(wave, samplerate)


def stack_array(
    arr: np.ndarray, buffer_size: int = INPUT_SAMPLES
) -> np.ndarray:
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
