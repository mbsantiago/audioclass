import datetime
from pathlib import Path
from typing import Callable, List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from soundevent import data


@pytest.fixture
def durations() -> List[float]:
    return [(index + 1) / 2 for index in range(20)]


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    return tmp_path / "audio"


@pytest.fixture
def file_list(
    audio_dir: Path,
    random_wav_factory,
    durations: List[float],
) -> List[Path]:
    return [
        random_wav_factory(
            path=audio_dir / f"{uuid4().hex}.wav",
            duration=duration,
            samplerate=48000,
        )
        for duration in durations
    ]


@pytest.fixture
def recordings(file_list: List[Path]) -> List[data.Recording]:
    return [data.Recording.from_file(path=path) for path in file_list]


@pytest.fixture
def dataframe(file_list):
    return pd.DataFrame(
        [
            {
                "path": path,
                "latitude": np.random.uniform(-90, 90),
                "longitude": np.random.uniform(-180, 180),
                "recorded_on": datetime.datetime(2021, 1, 1, 12, 0, 0)
                + datetime.timedelta(hours=np.random.randint(0, 24 * 365)),
                "site_id": f"site_{np.random.randint(0, 10)}",
            }
            for path in file_list
        ]
    )


@pytest.fixture
def process_array() -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    def process_array(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1 class and 1 feature
        return np.ones((array.shape[0], 1)), np.zeros((array.shape[0], 1))

    return process_array


@pytest.fixture
def tags() -> List[data.Tag]:
    return [data.Tag(term=data.term_from_key("test"), value="test")]
