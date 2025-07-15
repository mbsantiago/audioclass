from pathlib import Path
from typing import List

import pandas as pd
import pytest
from soundevent import data

from audioclass.batch.base import (
    recordings_from_dataframe,
    recordings_from_directory,
    recordings_from_files,
)


def test_from_dataframe_checks_columns(
    dataframe: pd.DataFrame,
):
    with pytest.raises(ValueError):
        recordings_from_dataframe(
            dataframe,
            path_col="foo",
        )


def test_recordings_from_dataframe_picks_up_additional_cols(
    dataframe: pd.DataFrame,
):
    recordings = recordings_from_dataframe(
        dataframe,
        additional_cols=["site_id"],
    )
    assert len(recordings) == len(dataframe)
    for recording in recordings:
        assert data.find_tag(recording.tags, "site_id") is not None


def test_can_run_get_recordings_from_files(
    file_list: List[Path],
):
    recordings = recordings_from_files(
        file_list,
    )
    assert len(recordings) == len(file_list)


def test_recordings_from_directory_skips_empty_files(
    file_list: List[Path],
    audio_dir: Path,
):
    (audio_dir / "empty_audio.wav").touch()

    recordings = recordings_from_directory(audio_dir)
    assert len(recordings) == len(file_list)


def test_can_get_all_audio_files_in_dir(
    random_wav_factory,
    tmp_path: Path,
):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    for index in range(10):
        random_wav_factory(path=audio_dir / f"audio_{index}.wav")

    for index in range(10):
        (audio_dir / f"text_{index}.txt").touch()

    recordings = recordings_from_directory(audio_dir)
    assert len(recordings) == 10
