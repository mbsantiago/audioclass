from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd
from soundevent import audio, data

from audioclass.constants import BATCH_SIZE

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

These indices are correspond to the index of the file in the list of files
being iterated over.
"""

FrameArray: TypeAlias = np.ndarray
"""A 1D array of frame indices of shape (batch_size, ).

These numbers correspond to the frame number in the audio file. In case an
audio file is split into multiple frames, this number will be the frame number.
"""

Batch: TypeAlias = Tuple[AudioArray, List[data.Recording], FrameArray]
"""A single batch of audio data."""


BatchGenerator: TypeAlias = Generator[
    Batch,
    None,
    None,
]
"""An iterator that yields batches of audio data."""


class BaseIterator(ABC):
    def __init__(
        self,
        recordings: List[data.Recording],
        samplerate: int,
        input_samples: int,
        batch_size: int = BATCH_SIZE,
        audio_dir: Optional[Path] = None,
    ):
        self.recordings = recordings
        self.samplerate = samplerate
        self.input_samples = input_samples
        self.batch_size = batch_size
        self.audio_dir = audio_dir

    @abstractmethod
    def __iter__(self) -> BatchGenerator: ...  # pragma: no cover

    @classmethod
    def from_files(
        cls,
        files: List[Path],
        samplerate: int,
        input_samples: int,
        batch_size: int = BATCH_SIZE,
        audio_dir: Optional[Path] = None,
    ):
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
        return cls(  # pragma: no cover
            recordings_from_directory(directory, recursive=recursive),
            samplerate=samplerate,
            input_samples=input_samples,
            batch_size=batch_size,
            audio_dir=directory,
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


def recordings_from_files(files: List[Path]) -> List[data.Recording]:
    return [
        data.Recording.from_file(file, compute_hash=False) for file in files
    ]


def recordings_from_directory(
    directory: Path,
    recursive: bool = True,
) -> List[data.Recording]:
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
        tags = [data.Tag(key=col, value=row[col]) for col in additional_cols]  # type: ignore
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
