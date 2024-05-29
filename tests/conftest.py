import random
import string
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import soundfile as sf


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def random_string():
    """Generate a random string of fixed length."""
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
    """Produce a random wav file."""

    def wav_factory(
        path: Optional[Path] = None,
        samplerate: int = 22100,
        duration: float = 0.1,
        channels: int = 1,
        bit_depth: Optional[int] = None,
        fmt: str = "WAV",
        subtype: Optional[str] = None,
    ) -> Path:
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
