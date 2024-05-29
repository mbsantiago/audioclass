"""Module for defining and providing audio classification models.

This module establishes the core interface for audio classification models
within the audioclass library. It defines the `ClipClassificationModel`
abstract base class, which serves as a blueprint for creating different model
implementations (e.g., TensorFlow, PyTorch, etc.). This ensures consistency in
how models are used and integrated into audio classification workflows.

The module also provides concrete implementations of audio classification
models, allowing users to directly apply them to their audio data. These models
adhere to the standardized interface defined by the `ClipClassificationModel`
class, making them easy to use and interchangeable within the library.

Additionally, this module defines the `ModelOutput` namedtuple, specifying the
standard format for the output produced by audio classification models. This
ensures that all models within the library produce results in a consistent and
predictable way, facilitating further analysis and post-processing.
"""

from audioclass.models.base import ClipClassificationModel, ModelOutput

__all__ = [
    "ClipClassificationModel",
    "ModelOutput",
]
