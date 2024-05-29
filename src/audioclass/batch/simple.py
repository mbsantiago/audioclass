from collections import deque

import numpy as np

from audioclass.batch.base import BaseIterator, BatchGenerator
from audioclass.preprocess import load_recording


class SimpleIterator(BaseIterator):
    def __iter__(self) -> BatchGenerator:
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
