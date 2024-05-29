"""Module for providing a simple audio batch iterator."""

from collections import deque

import numpy as np

from audioclass.batch.base import BaseIterator, BatchGenerator
from audioclass.preprocess import load_recording


class SimpleIterator(BaseIterator):
    """A straightforward iterator for processing audio recordings in batches.

    This iterator loads and preprocesses audio recordings one at a time, then
    groups them into batches for efficient processing by machine learning
    models.

    It's ideal for smaller datasets.

    Notes
    -----
    For larger datasets or situations requiring parallel processing, consider
    using alternative iterators like `TFDatasetIterator` or a custom
    implementation.
    """

    def __iter__(self) -> BatchGenerator:
        """Iterate over the audio data, yielding batches.

        Yields
        ------
        Batch
            A batch of audio data, consisting of:
            - A numpy array of shape (batch_size, num_samples) containing the
            audio data.
            - A list of corresponding `Recording` objects.
            - A numpy array of shape (batch_size,) containing the frame indices
            for each audio clip in the batch.
        """
        array_queue = deque(maxlen=self.batch_size * 4)
        recording_queue = deque(maxlen=self.batch_size * 4)
        frame_queue = deque(maxlen=self.batch_size * 4)

        for recording in self.recordings:
            array = load_recording(
                recording,
                samplerate=self.samplerate,
                buffer_size=self.input_samples,
                audio_dir=self.audio_dir,
            )

            array_queue.extend(array)
            recording_queue.extend([recording] * len(array))
            frame_queue.extend(range(len(array)))

            while len(array_queue) >= self.batch_size:
                array = np.array(
                    [array_queue.popleft() for _ in range(self.batch_size)]
                )
                recordings = [
                    recording_queue.popleft() for _ in range(self.batch_size)
                ]
                frames = np.array(
                    [frame_queue.popleft() for _ in range(self.batch_size)]
                )
                yield array, recordings, frames

        if len(array_queue) > 0:
            array = np.array(array_queue)
            recordings = list(recording_queue)
            frames = np.array(frame_queue)
            yield array, recordings, frames
