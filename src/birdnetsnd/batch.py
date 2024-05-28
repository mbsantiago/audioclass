from pathlib import Path
from typing import List, Optional

import pandas as pd
import tensorflow as tf
from soundevent import audio, data
from tensorflow import data as tfdata

from birdnetsnd.constants import DEFAULT_THRESHOLD, INPUT_SAMPLES, SAMPLERATE
from birdnetsnd.model import BirdNET
from birdnetsnd.preprocess import load_recording

__all__ = [
    "process_dataframe",
    "process_directory",
    "process_file_list",
]


def files_dataset(
    files: list[Path],
    batch_size: int = 32,
    samplerate: int = SAMPLERATE,
    input_samples: int = INPUT_SAMPLES,
    audio_dir: Optional[Path] = None,
) -> tfdata.Dataset:
    """Creates a TensorFlow Dataset for loading and processing audio files.

    This function takes a list of audio file paths and returns a
    `tfdata.Dataset` that yields batches of preprocessed audio data.  Each
    element of the dataset is a tuple containing:

    * A tensor of shape `(batch_size, input_samples)`, representing the audio
    waveform data.
    * A tuple of shape `(batch_size, 2)`, where the first element is the frame
    index (within the audio file) and the second element is the file index
    (within the list of files).

    Parameters
    ----------
    files : list[Path]
        A list of paths to the audio files to be included in the dataset. Paths
        can be absolute or relative to the `audio_dir` if provided, or the
        current working directory otherwise.
    batch_size : int, optional
        The number of audio samples to include in each batch. Default is 32.
    samplerate : int, optional
        The desired sampling rate for the audio data (e.g., 44100 Hz). Default
        is 48000 Hz. Any audio files with a different sampling rate will be
        resampled to match this value.
    input_samples : int, optional
        The number of samples to include in each audio clip (frame). Default is
        144000 samples (3 seconds at 48000 Hz). This should match the input
        size of your model.
    audio_dir : Optional[Path], optional
        An optional base directory for resolving relative file paths in `files`.

    Returns
    -------
    tfdata.Dataset
        A TensorFlow Dataset that yields batches of preprocessed audio data and
        their indices.

    Notes
    -----
    * Audio files are loaded with the specified `samplerate` and divided into
    non-overlapping frames of `input_samples` length.
    * If a file is shorter than `input_samples`, it will be zero-padded to
    match. If longer, it will be split into multiple frames.
    * The dataset is batched with the specified `batch_size` and prefetched for
    improved performance.

    Examples
    --------
    >>> audio_files = [
    ...     Path("file1.wav"),
    ...     Path("file2.wav"),
    ... ]  # Replace with your file paths
    >>> dataset = files_dataset(audio_files)
    >>> for batch_data, batch_indices in dataset:
    ...     process_audio(batch_data)
    """

    @tf.py_function(Tout=tf.float32)  # type: ignore
    def apply_fn(index):
        index = index.numpy()
        path = files[index]
        recording = data.Recording.from_file(path, compute_hash=False)
        return load_recording(
            recording,
            samplerate=samplerate,
            buffer_size=input_samples,
            audio_dir=audio_dir,
        )

    return (
        tfdata.Dataset.range(len(files))
        .map(
            apply_fn,
            name="load_recording",
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .enumerate()
        .flat_map(
            lambda ind, arr: tfdata.Dataset.zip(
                tfdata.Dataset.from_tensor_slices(arr, name="audio"),
                tfdata.Dataset.from_tensors([ind]).repeat().enumerate(),
                name="zip",
            )
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def process_recordings(
    model: BirdNET,
    recordings: List[data.Recording],
    batch_size: int = 32,
    audio_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_THRESHOLD,
) -> List[data.ClipPrediction]:
    files = [recording.path for recording in recordings]
    dataset = files_dataset(
        files,
        batch_size=batch_size,
        samplerate=model.samplerate,
        input_samples=model.input_samples,
        audio_dir=audio_dir,
    )
    hop_size = model.input_samples / model.samplerate

    output = []

    for array, (frame, index) in dataset.as_numpy_iterator():  # type: ignore
        class_probs, features = model.process_array(array)

        for probs, feats, frm, idx in zip(
            class_probs,
            features,
            frame,
            index,
        ):
            recording = recordings[idx]
            clip = data.Clip(
                recording=recording,
                start_time=frm * hop_size,
                end_time=(frm + 1) * hop_size,
            )
            output.append(
                data.ClipPrediction(
                    clip=clip,
                    features=[
                        data.Feature(name=f"{model.name}_{i}", value=feat)
                        for i, feat in enumerate(feats)
                    ],
                    tags=[
                        data.PredictedTag(tag=tag, score=score)
                        for tag, score in zip(model.tags, probs)
                        if score > confidence_threshold
                    ],
                )
            )

    return output


def process_file_list(
    model: BirdNET,
    files: List[Path],
    batch_size: int = 32,
):
    recordings = [
        data.Recording.from_file(file, compute_hash=False) for file in files
    ]
    return process_recordings(
        model,
        recordings,
        batch_size=batch_size,
    )


def process_directory(
    model: BirdNET,
    directory: Path,
    recursive: bool = True,
    batch_size: int = 32,
):
    files = list(audio.get_audio_files(directory, recursive=recursive))
    return process_file_list(
        model,
        files,
        batch_size=batch_size,
    )


def process_dataframe(
    model: BirdNET,
    df: pd.DataFrame,
    batch_size: int = 32,
    audio_dir: Optional[Path] = None,
    path_col: str = "path",
    latitude_col: Optional[str] = "latitude",
    longitude_col: Optional[str] = "longitude",
    recorded_on_col: Optional[str] = "recorded_on",
    additional_cols: Optional[list[str]] = None,
):
    if additional_cols is None:
        additional_cols = []

    cols = [
        path_col,
        latitude_col,
        longitude_col,
        recorded_on_col,
        *additional_cols,
    ]
    required_cols = [col for col in cols if col is not None]

    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    recordings = []
    for _, row in df.iterrows():
        path = Path(row[path_col])  # type: ignore
        recorded_on = row.get(recorded_on_col)
        recording = data.Recording.from_file(
            path=path,
            compute_hash=False,
            latitude=row.get(latitude_col),
            longitude=row.get(longitude_col),
            date=recorded_on.date() if recorded_on else None,
            time=recorded_on.time() if recorded_on else None,
            **{col: row[col] for col in additional_cols},  # type: ignore
        )
        recordings.append(recording)

    return process_recordings(
        model,
        recordings,
        batch_size=batch_size,
        audio_dir=audio_dir,
    )
