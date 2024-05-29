from pathlib import Path
from typing import List

import numpy as np
from audioclass.batch import SimpleIterator
from soundevent import data


def test_simple_iterator(file_list: List[Path]):
    iterator = SimpleIterator.from_files(
        file_list,
        samplerate=48000,
        input_samples=144000,
        batch_size=4,
    )
    for array, recordings, frame in iterator:
        assert isinstance(array, np.ndarray)
        assert isinstance(recordings, list)
        assert isinstance(frame, np.ndarray)
        assert tuple(array.shape) == (4, 144000)
        assert tuple(frame.shape) == (4,)
        assert all(isinstance(rec, data.Recording) for rec in recordings)


def test_simple_iterator_goes_through_all_frames(
    file_list: List[Path],
    durations: List[float],
):
    samplerate = 48000
    input_samples = 144000
    batch_size = 3
    hop_size = input_samples / samplerate

    iterator = SimpleIterator.from_files(
        file_list,
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
