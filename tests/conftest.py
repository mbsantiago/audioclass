import random
import string
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import soundfile as sf


def random_string() -> str:
    """Generate a random string of fixed length.

    Returns
    -------
    str
        A random string of 10 characters.
    """
    options = string.ascii_uppercase + string.digits
    return "".join(random.choice(options) for _ in range(10))


def write_random_wave(
    path: Path,
    samplerate: int = 22100,
    duration: float = 0.1,
    channels: int = 1,
    fmt: str = "WAV",
    subtype: Optional[str] = None,
):
    """Write a random wave file to disk."""
    frames = int(samplerate * duration)
    shape = (frames, channels)
    wav = np.random.random(size=shape)
    sf.write(path, wav, samplerate, format=fmt, subtype=subtype)


@pytest.fixture
def random_wav_factory(tmp_path: Path):
    """Produce a random audio file.

    Returns
    -------
    Callable
        A function that creates a random audio file.
    """

    def wav_factory(
        path: Optional[Path] = None,
        samplerate: int = 22100,
        duration: float = 0.1,
        channels: int = 1,
        bit_depth: Optional[int] = None,
        fmt: str = "WAV",
        subtype: Optional[str] = None,
    ) -> Path:
        """Create a random audio file.

        Parameters
        ----------
        path
            The path to save the audio file. If None, a temporary path will be
            created.
        samplerate
            The sample rate of the audio file.
        duration
            The duration of the audio file in seconds.
        channels
            The number of channels in the audio file.
        bit_depth
            The number of bits per sample.
        fmt
            The format of the audio file.
        subtype
            The subtype of the audio file.

        Returns
        -------
        Path
            The path to the saved audio file.
        """
        if path is None:
            path = tmp_path / (random_string() + ".wav")

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        if bit_depth is not None:
            fmt = "WAV"
            subtype = f"PCM_{bit_depth}"

        write_random_wave(
            path=path,
            samplerate=samplerate,
            duration=duration,
            channels=channels,
            fmt=fmt,
            subtype=subtype,
        )

        return path

    return wav_factory
