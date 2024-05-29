from typing import List

import numpy as np
from audioclass.batch.tensorflow import TFDatasetIterator
from soundevent import data


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
