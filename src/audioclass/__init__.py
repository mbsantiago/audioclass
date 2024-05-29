"""Audioclass: A Python library for audio classification.

The audioclass library provides a unified framework for working with audio
classification models, regardless of their underlying implementation
(TensorFlow, TensorFlow Lite, etc.). It simplifies the process of loading,
preprocessing, batching, and analyzing audio data, enabling seamless
integration with various machine learning models.

Key Features:

* **Standardized Model Interface:** The `ClipClassificationModel` abstract base
class defines a consistent interface for all models, making them
interchangeable and easy to use.
* **Flexible Data Loading:** Load audio data from files, directories, or pandas
DataFrames, with support for resampling and frame-based processing.
* **Efficient Batch Processing:** Process large audio datasets in batches using
optimized iterators like `SimpleIterator` and `TFDatasetIterator`.
* **Unified Postprocessing:** Convert model outputs into xarray datasets or
lists of soundevent objects for further analysis and visualization.
* **Pre-trained Models:** Includes convenient access to popular audio
classification models like BirdNET and Perch.

Module Structure:

The audioclass library is organized into the following modules:

* `audioclass.models`: Defines the model interface (`ClipClassificationModel`)
and provides concrete model implementations.
* `audioclass.preprocess`: Provides functions for loading and preprocessing
audio data.
* `audioclass.batch`: Offers tools for batch processing audio data, including
iterators and a processing function.
* `audioclass.postprocess`: Includes functions for converting model outputs
into different formats.
* `audioclass.utils`: Contains various utility functions for working with audio
data and models.
"""

from audioclass.batch import (
    BaseIterator,
    BatchGenerator,
    SimpleIterator,
    process_iterable,
)
from audioclass.models import ClipClassificationModel, ModelOutput
from audioclass.preprocess import (
    load_clip,
    load_recording,
)

__all__ = [
    "BaseIterator",
    "BatchGenerator",
    "ClipClassificationModel",
    "ClipClassificationModel",
    "ModelOutput",
    "ModelOutput",
    "SimpleIterator",
    "load_clip",
    "load_recording",
    "process_iterable",
]
