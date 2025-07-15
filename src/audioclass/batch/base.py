"""Module for defining base classes and functions for batch processing.

This module provides abstract classes and utility functions for creating
iterators that generate batches of audio data from various sources, such as
lists of files, directories, or pandas DataFrames. These iterators are designed
to be used with audio classification models to process large amounts of audio
data efficiently.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from soundevent import audio, data

from audioclass.constants import BATCH_SIZE

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

__all__ = [
    "Batch",
    "BatchGenerator",
    "BaseIterator",
    "recordings_from_files",
    "recordings_from_directory",
    "recordings_from_dataframe",
]

AudioArray: TypeAlias = np.ndarray
"""A 2D array of audio samples of shape (batch_size, n_samples)."""

IndexArray: TypeAlias = np.ndarray
"""A 1D array of indices of shape (batch_size, ).

These indices correspond to the index of the file in the list of files
being iterated over.
"""

FrameArray: TypeAlias = np.ndarray
"""A 1D array of frame indices of shape (batch_size, ).

These numbers correspond to the frame number in the audio file. In case an
audio file is split into multiple frames, this number will be the frame number.
"""

Batch: TypeAlias = Tuple[AudioArray, List[data.Recording], FrameArray]
"""A single batch of audio data consisting of:

- AudioArray: The audio data as a numpy array.
- List[data.Recording]: The corresponding list of Recording objects.
- FrameArray: The frame indices for each audio clip in the batch.
"""


BatchGenerator: TypeAlias = Generator[
    Batch,
    None,
    None,
]
"""A generator that yields batches of audio data."""


class BaseIterator(ABC):
    """Abstract base class for audio batch iterators.

    This class defines the common interface for iterators that generate batches
    of audio data from different sources. It provides methods for creating
    iterators from files, directories, and pandas DataFrames.
    """

    recordings: List[data.Recording]
    """The list of Recording objects to be processed."""

    samplerate: int
    """The target sample rate for resampling the audio data (in Hz)."""

    input_samples: int
    """The number of samples per audio frame."""

    batch_size: int
    """The number of audio frames per batch."""

    audio_dir: Optional[Path]
    """The directory containing the audio files."""

    def __init__(
        self,
        recordings: List[data.Recording],
        samplerate: int,
        input_samples: int,
        batch_size: int = BATCH_SIZE,
        audio_dir: Optional[Path] = None,
    ):
        """Initialize the BaseIterator.

        Parameters
        ----------
        recordings
            A list of `Recording` objects representing the audio files to be
            processed.
        samplerate
            The target sample rate for resampling the audio data (in Hz).
        input_samples
            The number of samples per audio frame.
        batch_size
            The number of audio frames per batch. Defaults to `BATCH_SIZE`.
        audio_dir
            The directory containing the audio files. Defaults to None.
        """
        self.recordings = recordings
        self.samplerate = samplerate
        self.input_samples = input_samples
        self.batch_size = batch_size
        self.audio_dir = audio_dir

    @abstractmethod
    def __iter__(self) -> BatchGenerator:  # pragma: no cover
        """Iterate over the audio data, yielding batches.

        This is an abstract method that must be implemented by subclasses.

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

    @classmethod
    def from_files(
        cls,
        files: List[Path],
        samplerate: int,
        input_samples: int,
        batch_size: int = BATCH_SIZE,
        audio_dir: Optional[Path] = None,
    ):
        """Create a batch iterator from a list of audio files.

        Parameters
        ----------
        files
            A list of paths to the audio files.
        samplerate
            The target sample rate for resampling the audio data (in Hz).
        input_samples
            The number of samples per audio frame.
        batch_size
            The number of audio frames per batch. Defaults to `BATCH_SIZE`.
        audio_dir
            The directory containing the audio files. Defaults to None.

        Returns
        -------
        BaseIterator
            A batch iterator for the specified audio files.
        """
        return cls(  # pragma: no cover
            recordings_from_files(files),
            samplerate=samplerate,
            input_samples=input_samples,
            batch_size=batch_size,
            audio_dir=audio_dir,
        )

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        samplerate: int,
        input_samples: int,
        batch_size: int = BATCH_SIZE,
        recursive: bool = True,
    ):
        """Create a batch iterator from a directory of audio files.

        Parameters
        ----------
        directory
            The path to the directory containing the audio files.
        samplerate
            The target sample rate for resampling the audio data (in Hz).
        input_samples
            The number of samples per audio frame.
        batch_size
            The number of audio frames per batch. Defaults to `BATCH_SIZE`.
        recursive
            Whether to search for audio files recursively in subdirectories.
            Defaults to True.

        Returns
        -------
        BaseIterator
            A batch iterator for the audio files in the specified directory.
        """
        return cls(  # pragma: no cover
            recordings_from_directory(directory, recursive=recursive),
            samplerate=samplerate,
            input_samples=input_samples,
            batch_size=batch_size,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        samplerate: int,
        input_samples: int,
        batch_size: int = BATCH_SIZE,
        audio_dir: Optional[Path] = None,
        path_col: str = "path",
        latitude_col: Optional[str] = "latitude",
        longitude_col: Optional[str] = "longitude",
        recorded_on_col: Optional[str] = "recorded_on",
        additional_cols: Optional[list[str]] = None,
    ):
        """Create a batch iterator from a pandas DataFrame.

        Parameters
        ----------
        df
            A DataFrame containing information about the audio files.
        samplerate
            The target sample rate for resampling the audio data (in Hz).
        input_samples
            The number of samples per audio frame.
        batch_size
            The number of audio frames per batch. Defaults to `BATCH_SIZE`.
        audio_dir
            The directory containing the audio files. Defaults to None.
        path_col
            The name of the column in the DataFrame containing the paths to the
            audio files. Defaults to "path".
        latitude_col
            The name of the column in the DataFrame containing the latitudes of
            the recording locations. Defaults to "latitude".
        longitude_col
            The name of the column in the DataFrame containing the longitudes
            of the recording locations. Defaults to "longitude".
        recorded_on_col
            The name of the column in the DataFrame containing the recording
            timestamps. Defaults to "recorded_on".
        additional_cols
            A list of additional columns in the DataFrame to include as tags in
            the `Recording` objects. Defaults to None.

        Returns
        -------
        BaseIterator
            A batch iterator for the audio files specified in the DataFrame.
        """
        return cls(  # pragma: no cover
            recordings_from_dataframe(
                df,
                path_col=path_col,
                latitude_col=latitude_col,
                longitude_col=longitude_col,
                recorded_on_col=recorded_on_col,
                additional_cols=additional_cols,
            ),
            samplerate=samplerate,
            input_samples=input_samples,
            batch_size=batch_size,
            audio_dir=audio_dir,
        )


def recordings_from_files(
    files: List[Path],
    ignore_errors: bool = True,
) -> List[data.Recording]:
    """Create a list of `Recording` objects from a list of file paths.

    This function iterates through a list of file paths, creating a
    `soundevent.data.Recording` object for each file. It can optionally
    ignore errors that occur during file processing.

    Parameters
    ----------
    files
        A list of `pathlib.Path` objects pointing to audio files.
    ignore_errors
        If True, any errors encountered while creating a `Recording` from a
        file will be ignored, and the file will be skipped. If False, any
        error will be raised. Defaults to True.

    Returns
    -------
    List[data.Recording]
        A list of `Recording` objects created from the provided files.

    Raises
    ------
    Exception
        If `ignore_errors` is False and an error occurs while processing a
        file.
    """
    recordings = []

    for path in files:
        try:
            recording = data.Recording.from_file(path, compute_hash=False)
            recordings.append(recording)
        except Exception as e:
            if not ignore_errors:
                raise e
            continue

    return recordings


def recordings_from_directory(
    directory: Path,
    recursive: bool = True,
) -> List[data.Recording]:
    """Create a list of `Recording` objects from audio files in a directory.

    Parameters
    ----------
    directory
        The path to the directory containing the audio files.
    recursive
        Whether to search for audio files recursively in subdirectories.
        Defaults to True.

    Returns
    -------
    List[data.Recording]
        A list of `Recording` objects corresponding to the audio files found in
        the directory.
    """
    files = list(audio.get_audio_files(directory, recursive=recursive))
    return recordings_from_files(files)


def recordings_from_dataframe(
    df: pd.DataFrame,
    path_col: str = "path",
    latitude_col: Optional[str] = "latitude",
    longitude_col: Optional[str] = "longitude",
    recorded_on_col: Optional[str] = "recorded_on",
    additional_cols: Optional[list[str]] = None,
) -> List[data.Recording]:
    """Create a list of `Recording` objects from a pandas DataFrame.

    The DataFrame should contain a column with file paths (specified by
    `path_col`), and optionally columns for latitude, longitude, recorded_on
    timestamp, and any additional columns to be included as tags in the
    `Recording` objects.

    Parameters
    ----------
    df
        A pandas DataFrame containing information about the audio files.
    path_col
        The name of the column containing the file paths. Defaults to "path".
    latitude_col
        The name of the column containing the latitudes. Defaults to
        "latitude".
    longitude_col
        The name of the column containing the longitudes. Defaults to
        "longitude".
    recorded_on_col
        The name of the column containing the recorded_on timestamps. Defaults
        to "recorded_on".
    additional_cols
        A list of additional column names to include as tags in the `Recording`
        objects. Defaults to None.

    Returns
    -------
    List[data.Recording]
        A list of `Recording` objects corresponding to the rows in the
        DataFrame.

    Raises
    ------
    ValueError
        If the DataFrame does not contain the required columns.
    """
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
        tags = [
            data.Tag(term=data.term_from_key(col), value=row[col])  # type: ignore
            for col in additional_cols
        ]
        recording = data.Recording.from_file(
            path=path,
            compute_hash=False,
            latitude=row.get(latitude_col),
            longitude=row.get(longitude_col),
            date=recorded_on.date() if recorded_on else None,
            time=recorded_on.time() if recorded_on else None,
            tags=tags,
        )
        recordings.append(recording)

    return recordings
