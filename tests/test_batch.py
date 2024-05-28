import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from birdnetsnd.batch import files_dataset, process_dataframe
from birdnetsnd.model import BirdNET


@pytest.fixture
def file_list(random_wav_factory) -> List[Path]:
    return [
        random_wav_factory(
            duration=(index + 1) / 2,
            samplerate=48000,
        )
        for index in range(20)
    ]


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
def birdnet_model():
    return BirdNET.from_model_file()


def test_process_dataframe_checks_columns(dataframe, birdnet_model):
    with pytest.raises(ValueError):
        process_dataframe(birdnet_model, dataframe, path_col="foo")


def test_can_iterate_over_file_list(file_list: List[Path]):
    dataset = files_dataset(
        file_list,
        batch_size=4,
        input_samples=144000,
    )

    for array, (frame, index) in dataset.as_numpy_iterator():  # type: ignore
        assert isinstance(array, np.ndarray)
        assert isinstance(index, np.ndarray)
        assert isinstance(frame, np.ndarray)
        assert tuple(array.shape) == (4, 144000)
        assert tuple(index.shape) == (4, 1)
        assert tuple(frame.shape) == (4,)
