"""Module for batch processing audio data for classification tasks.

This module provides tools for processing large collections of audio recordings
or clips in batches. It offers:

1. **Batch Iterators:** Classes like
   [`SimpleIterator`][audioclass.batch.SimpleIterator] and
   [`TFDatasetIterator`][audioclass.batch.tensorflow.TFDatasetIterator]
   generate batches of audio data from various sources (files, directories,
                                                        DataFrames).

2. **Batch Processing Function:** The
   [`process_iterable`][audioclass.batch.process_iterable] function applies a
   model's processing function to each batch of audio data, streamlining the
   classification workflow.

"""

from audioclass.batch.base import (
    BaseIterator,
    Batch,
    BatchGenerator,
    recordings_from_dataframe,
    recordings_from_directory,
    recordings_from_files,
)
from audioclass.batch.process import process_iterable
from audioclass.batch.simple import SimpleIterator

__all__ = [
    "Batch",
    "BaseIterator",
    "BatchGenerator",
    "SimpleIterator",
    "process_iterable",
    "recordings_from_directory",
    "recordings_from_files",
    "recordings_from_dataframe",
]
