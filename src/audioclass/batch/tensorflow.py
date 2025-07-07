"""Module for providing a TensorFlow Dataset-based audio batch iterator."""

import tensorflow as tf
from tensorflow import data as tfdata

from audioclass.batch.base import BaseIterator, BatchGenerator
from audioclass.preprocess import load_recording

__all__ = [
    "TFDatasetIterator",
]


class TFDatasetIterator(BaseIterator):
    """A TensorFlow Dataset-based audio batch iterator.

    This iterator leverages TensorFlow's Dataset API to efficiently load,
    preprocess, and batch audio data for model training and inference. It
    provides parallel loading and preprocessing capabilities, making it
    suitable for large datasets.
    """

    def __iter__(self) -> BatchGenerator:
        """Iterate over the audio data, yielding batches.

        Yields
        ------
        Batch
            A batch of audio data, consisting of:
                - A numpy array of shape (batch_size, num_samples) containing
                the audio data.
                - A list of corresponding `Recording` objects.
                - A numpy array of shape (batch_size,) containing the frame
                indices for each audio clip in the batch.
        """

        @tf.py_function(Tout=tf.float32)  # type: ignore
        def apply_fn(index):  # pragma: no cover
            index = index.numpy()
            recording = self.recordings[index]
            return load_recording(
                recording,
                samplerate=self.samplerate,
                buffer_size=self.input_samples,
                audio_dir=self.audio_dir,
            )

        dataset = (
            tfdata.Dataset.range(len(self.recordings))
            .map(
                apply_fn,
                name="load_recording",
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .ignore_errors()
            .enumerate()
            .flat_map(
                lambda ind, arr: tfdata.Dataset.zip(
                    tfdata.Dataset.from_tensor_slices(arr, name="audio"),
                    tfdata.Dataset.from_tensors([ind]).repeat().enumerate(),
                    name="zip",
                )
            )
            .batch(self.batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        for batch_data, (frame, index) in dataset.as_numpy_iterator():  # type: ignore
            yield (
                batch_data,
                [self.recordings[int(i)] for i in index[:, 0]],
                frame,
            )
