"""Utility functions for audioclass.

This module provides various helper functions for working with audio data and
models.
"""

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import BinaryIO, Generator, Optional, Union
from urllib.parse import urlparse

import numpy as np
import requests

__all__ = [
    "flat_sigmoid",
    "load_artifact",
    "batched",
]

DEFAULT_TIMEOUT: float = 30
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.25


def _hash_url(url: str) -> str:
    """Calculate the SHA256 hash of a URL.

    Parameters
    ----------
    url
        The URL to hash.

    Returns
    -------
    str
        The SHA256 hash of the URL.
    """
    return hashlib.sha256(url.encode()).hexdigest()


def _get_cache_dir() -> Path:
    """Get the cache directory for audioclass.

    This function returns the path to the cache directory, which is determined
    by the `XDG_CACHE_HOME` environment variable if it is set, otherwise it
    defaults to a temporary directory.

    Returns
    -------
    Path
        The path to the cache directory.
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")

    if xdg_cache_home:
        return Path(xdg_cache_home) / "audioclass"

    return Path(tempfile.gettempdir()) / "audioclass"


def _is_url(path: str) -> bool:
    """Check if a path is a URL.

    Parameters
    ----------
    path
        The path to check.

    Returns
    -------
    bool
        True if the path is a URL, False otherwise.
    """
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"} and parsed.netloc != ""


def _sha256_file(path: Path) -> str:
    """Calculate the SHA256 hash of a file.

    Parameters
    ----------
    path
        The file to hash.

    Returns
    -------
    str
        The SHA256 digest in hexadecimal format.
    """
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_artifact(
    path: Union[Path, str],
    directory: Optional[Path] = None,
    download: bool = True,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
) -> Path:
    """Load an artifact from a local path or a URL.

    If the path is a URL, the artifact is downloaded and cached in a local
    directory. If the artifact is already cached, it is loaded from the cache.

    Parameters
    ----------
    path
        The path to the artifact, either a local path or a URL.
    directory
        The directory to cache the artifact in. If not provided, a default
        cache directory is used.
    download
        Whether to download the artifact if it is not found in the cache.
        Defaults to True.
    timeout
        Timeout in seconds for each HTTP request. Defaults to 30.
    retries
        Maximum number of HTTP attempts for transient failures. Defaults to 3.
    backoff_factor
        Multiplier used to compute exponential backoff between retries.
        Defaults to 0.25.
    expected_sha256
        Expected SHA256 checksum of the downloaded artifact. If provided,
        mismatch raises ValueError.
    expected_size
        Expected file size in bytes of the downloaded artifact. If provided,
        mismatch raises ValueError.

    Returns
    -------
    Path
        The path to the loaded artifact.

    Raises
    ------
    FileNotFoundError
        If the artifact is not found and `download` is False.
    """
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
        try:
            _validate_artifact(
                new_path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            return new_path
        except ValueError:
            if not download:
                raise
            new_path.unlink()

    if not download:
        raise FileNotFoundError(
            f"Could not find artifact at {new_path} corresponding to {path}"
            f" and download is disabled."
        )

    try:
        with new_path.open("wb") as f:
            _download_with_retry(
                path,
                f,
                timeout=timeout,
                retries=retries,
                backoff_factor=backoff_factor,
            )
    except Exception:
        if new_path.exists():
            new_path.unlink()
        raise

    try:
        _validate_artifact(
            new_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    except ValueError:
        if new_path.exists():
            new_path.unlink()
        raise

    return new_path


def _download_with_retry(
    url: str,
    fileobj: BinaryIO,
    timeout: float,
    retries: int,
    backoff_factor: float,
) -> None:
    if retries < 1:
        raise ValueError("retries must be at least one")

    if backoff_factor < 0:
        raise ValueError("backoff_factor must be non-negative")

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fileobj.write(chunk)

            return
        except requests.RequestException as error:
            last_error = error

            if attempt == retries:
                raise

            delay = backoff_factor * (2 ** (attempt - 1))
            time.sleep(delay)

    if last_error is not None:  # pragma: no cover
        raise last_error


def _validate_artifact(
    path: Path,
    expected_sha256: Optional[str],
    expected_size: Optional[int],
) -> None:
    if expected_size is not None:
        size = path.stat().st_size
        if size != expected_size:
            raise ValueError(
                f"Invalid file size for artifact at {path}"
                f" ({size} != {expected_size})"
            )

    if expected_sha256 is not None:
        digest = _sha256_file(path)
        if digest.lower() != expected_sha256.lower():
            raise ValueError(
                f"Invalid SHA256 checksum for artifact at {path}"
                f" ({digest} != {expected_sha256})"
            )


def flat_sigmoid(
    x: np.ndarray,
    sensitivity: float = 1,
    vmin: float = -15,
    vmax: float = 15,
) -> np.ndarray:
    """Apply a flattened sigmoid function to an array.

    This function applies a sigmoid function to each element of the
    input array, but with a flattened shape to prevent extreme values.
    The output values are clipped between 0 and 1.

    Parameters
    ----------
    x
        The input array.
    sensitivity
        The sensitivity of the sigmoid function. Defaults to 1.
    vmin
        The minimum value to clip the input to. Defaults to -15.
    vmax
        The maximum value to clip the input to. Defaults to 15.

    Returns
    -------
    np.ndarray
        The output array with the same shape as the input.
    """
    return 1 / (1 + np.exp(-sensitivity * np.clip(x, vmin, vmax)))


def batched(
    array: np.ndarray,
    n: int,
    axis: int = 0,
    *,
    strict: bool = False,
) -> Generator[np.ndarray, None, None]:
    """Batch array data along an axis.

    This function yields batches of data from an array along a specified axis.
    The final batch may be shorter than the specified batch size if the array
    length is not divisible by n.

    Parameters
    ----------
    array
        The input array.
    n
        The batch size.
    strict
        Whether to raise an error if the final batch is shorter than n.

    Yields
    ------
    batch : np.ndarray
        The next batch of data from the array.

    Raises
    ------
    ValueError
        If n is less than 1 or if strict is True and the final batch is shorter
        than n.
    """
    if n < 1:
        raise ValueError("n must be at least one")

    shape = list(array.shape)
    length = shape[axis]

    for i in range(0, length, n):
        s = tuple(
            slice(None) if j != axis else slice(i, i + n)
            for j in range(len(shape))
        )

        batch = array[s]

        if strict and batch.shape[axis] < n:
            raise ValueError("Final batch is shorter than n")

        yield batch
