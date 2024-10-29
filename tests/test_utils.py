from pathlib import Path

import pytest

from audioclass.models.birdnet import LABELS_PATH, MODEL_PATH
from audioclass.utils import _hash_url, _is_url, load_artifact


def test_can_detect_birdnet_urls():
    assert _is_url(MODEL_PATH)
    assert _is_url(LABELS_PATH)


def test_load_artifact(tmp_path: Path):
    directory = tmp_path / "test"
    path = load_artifact(LABELS_PATH, directory=directory)
    assert path.exists()
    assert path == directory / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"


def test_load_artifact_does_nothing_if_not_a_url(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.touch()
    path = load_artifact(str(artifact))
    assert path == artifact


def test_load_artifact_fails_if_download_disabled(tmp_path: Path):
    directory = tmp_path / "test"
    with pytest.raises(FileNotFoundError):
        load_artifact(LABELS_PATH, directory=directory, download=False)


def test_load_artifact_does_not_download_if_file_exists(
    tmp_path: Path,
    mocker,
):
    directory = tmp_path / "test"
    directory.mkdir()
    path = directory / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
    path.touch()

    request_get = mocker.patch("requests.get")
    path = load_artifact(LABELS_PATH, directory=directory)
    request_get.assert_not_called()


def test_loaded_artifact_is_stored_in_tempdir(
    tmp_path: Path,
    monkeypatch,
):
    tempdir = tmp_path / "tempdir"
    tempdir.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tempdir))

    path = load_artifact(LABELS_PATH)
    assert path.exists()
    assert path.parent.parent == tempdir / "audioclass"
    assert path.parent.name == _hash_url(LABELS_PATH)
