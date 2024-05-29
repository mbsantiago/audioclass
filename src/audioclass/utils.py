import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np
import requests

__all__ = [
    "flat_sigmoid",
    "load_artifact",
]


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _get_cache_dir() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")

    if xdg_cache_home:
        return Path(xdg_cache_home) / "audioclass"

    return Path(tempfile.gettempdir()) / "audioclass"


def _is_url(path: str) -> bool:
    return urlparse(path).scheme != ""


def load_artifact(
    path: Union[Path, str],
    directory: Optional[Path] = None,
    download: bool = True,
) -> Path:
    if isinstance(path, Path) or not _is_url(path):
        return Path(path)

    if directory is None:
        hash = _hash_url(path)
        directory = _get_cache_dir() / hash

    if not directory.exists():
        directory.mkdir(parents=True)

    basename = Path(urlparse(path).path).name
    new_path = directory / basename

    if new_path.exists():
        return new_path

    if not download:
        raise FileNotFoundError(
            f"Could not find artifact at {new_path} corresponding to {path}"
            f" and download is disabled."
        )

    with new_path.open("wb") as f:
        response = requests.get(path)
        response.raise_for_status()
        f.write(response.content)

    return new_path


def flat_sigmoid(
    x: np.ndarray,
    sensitivity: float = 1,
    vmin: float = -15,
    vmax: float = 15,
) -> np.ndarray:
    return 1 / (1 + np.exp(-sensitivity * np.clip(x, vmin, vmax)))
