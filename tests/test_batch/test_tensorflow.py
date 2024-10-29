from pathlib import Path
from typing import List

import pytest

pytest.importorskip("tensorflow")

import numpy as np
from soundevent import data

from audioclass.batch.tensorflow import TFDatasetIterator


@pytest.mark.tensorflow
def test_tf_iterator(recordings: List[data.Recording]):
    iterator = TFDatasetIterator(
        recordings,
        samplerate=48000,
        input_samples=144000,
        batch_size=4,
    )
    for array, recordings, frame in iterator:
        assert array.shape == (4, 144000)
        assert frame.shape == (4,)
        assert len(recordings) == 4
        assert all(isinstance(rec, data.Recording) for rec in recordings)


@pytest.mark.tensorflow
def test_tf_iterator_goes_through_all_frames(
    recordings: List[data.Recording],
    durations: List[float],
):
    samplerate = 48000
    input_samples = 144000
    batch_size = 3
    hop_size = input_samples / samplerate

    iterator = TFDatasetIterator(
        recordings,
        samplerate=samplerate,
        input_samples=input_samples,
        batch_size=batch_size,
    )

    total_frames = sum(
        int(np.ceil(duration / hop_size)) for duration in durations
    )

    frames = 0
    for array, _, _ in iterator:
        frames += array.shape[0]

    assert frames == total_frames


@pytest.mark.tensorflow
def test_can_iterate_over_files_in_directory(
    audio_dir: Path,
    file_list: List[Path],
):
    iterator = TFDatasetIterator.from_directory(
        audio_dir,
        samplerate=48000,
        input_samples=144000,
        batch_size=4,
    )

    assert len(iterator.recordings) == len(file_list)

    for array, recordings, frame in iterator:
        assert isinstance(array, np.ndarray)
        assert isinstance(recordings, list)
        assert isinstance(frame, np.ndarray)
        assert tuple(array.shape) == (4, 144000)
        assert all(isinstance(rec, data.Recording) for rec in recordings)
        assert all(rec.path in file_list for rec in recordings)
        assert all(rec.path.exists() for rec in recordings)
