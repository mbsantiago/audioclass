# Usage Guide

## Introduction

**audioclass** is a Python library designed to simplify audio classification
tasks using machine learning. It provides a standardized interface for working
with various audio classification models, regardless of their underlying
implementation. audioclass streamlines the process of loading, preprocessing,
and analyzing audio data, enabling easy integration with diverse machine
learning models and making audio classification more accessible.

## Model List

**audioclass** includes built-in support for two popular audio classification
models:

- [**BirdNET:**][audioclass.models.birdnet.BirdNET] A TensorFlow Lite model
  specifically designed for bird sound classification. BirdNET can identify a
  wide range of bird species and is suitable for biodiversity monitoring and
  ecological research.

- [**BirdNETAnalyzer:**][audioclass.models.birdnet_analyzer.BirdNETAnalyzer] A
  TensorFlow-based model designed for bird sound classification, offering
  potential GPU acceleration. BirdNETAnalyzer provides a similar functionality to
  the existing BirdNET model but is optimized for scenarios where computational
  resources are abundant and inference speed is a priority.

- [**Perch:**][audioclass.models.perch.Perch] A TensorFlow Hub model developed
  by Google Research for bird sound classification. Perch is trained on a large
  dataset of bird vocalizations and is known for its quality embeddings.

## Using the Models

Working with audio classification models in `audioclass` is simple and intuitive.
All models share a consistent interface, making it easy to switch between them
without significant code changes.

You have several options for processing audio data, depending on your starting point.
Let's look at how you can process audio data using the BirdNET model:

```python
from audioclass.models.birdnet import BirdNET
from soundevent import data

model = BirdNET.load()
recording = data.Recording.from_file("path/to/your/audio.wav")
predictions = model.process_recording(recording)

# Explore predictions
for clip_prediction in predictions:
    print(
        clip_prediction.clip.start_time,
        clip_prediction.clip.end_time,
        clip_prediction.tags,
    )
```

In addition to processing whole recordings, `audioclass` offers flexibility to
work with different types of audio data:

### Process a File Directly

If you don't need the extra information from a Recording object, you can pass the file path directly:

```python
predictions = model.process_file("path/to/your/audio.wav")
```

### Process a Clip

Isolate a specific segment of a recording for analysis:

```python
clip = data.Clip(start_time=10, end_time=20, recording=recording)
predictions = model.process_clip(clip)
```

### Process Raw Audio Data

For maximum control, work with raw audio arrays (NumPy arrays):

```python
import numpy as np
raw_audio = np.random.randn(44100)  # Random audio (replace with your data)
predictions = model.process_array(raw_audio)
```

### Interchangeable Models:

All models in `audioclass` adhere to the same interface, making it effortless to switch between them:

```python
from audioclass.models.perch import Perch  # Switch to the Perch model

model = Perch.load()
# ... the rest of your code remains the same!
```

### Model Attributes

Each model provides essential information through its attributes:

- `model.samplerate`: The sample rate the model expects (e.g., 48000 Hz for BirdNET).
- `model.input_samples`: The number of samples per audio frame the model requires.
- `model.tags`: The labels the model can predict.

## Model Outputs

`audioclass` models provide results as a list of [`ClipPrediction`][soundevent.data.ClipPrediction] objects from
the [`soundevent`][soundevent] library. These objects offer a convenient, standardized way to
access predictions and features for each audio clip processed.

### What's Inside a `ClipPrediction`?

Each [`ClipPrediction`][soundevent.data.ClipPrediction] object holds essential information:

- **Clip Details:**
  - The `clip` attribute provides the audio clip's start and end times within the original recording.
  - The `recording` attribute gives you access to details about the original recording (e.g., location, date, etc.).
- **Predicted Tags:** The `tags` attribute is a list of [`PredictedTag`][soundevent.data.PredictedTag] objects, each representing a class label the model thinks is present in the clip. Each `PredictedTag` has:
  - A `tag` attribute for the class label.
  - A `score` attribute indicating the model's confidence in that label (typically a probability between 0 and 1).
- **Extracted Features:** The `features` attribute contains a list of [`Feature`][soundevent.data.Feature] objects, representing numerical representations extracted from the audio by the model. These features can be used for further analysis, clustering, or even as input for other machine learning tasks.

### Working with `ClipPrediction`

```python
outputs = model.process_recording(recording)

clip_prediction = outputs[0] # access the 1st prediction

# Get the audio clip information
clip = clip_prediction.clip
recording = clip.recording
start_time = clip.start_time
end_time = clip.end_time

# Access predicted tags
for predicted_tag in clip_prediction.tags:
    tag = predicted_tag.tag
    score = predicted_tag.score
    print(f"The tag '{tag}' was predicted to be present in the clip with {score:.2f} confidence.")

# Access extracted features
for feature in clip_prediction.features:
    print(f"Extracted feature: {feature.name} - {feature.value}")
```

As mentioned, the `soundevent` package gives a convenient way of working with
model outputs. For a full description of these objects please refer to the
`soundevent` [documentation](https://mbsantiago.github.io/soundevent/).

## Batch Processing

Working with large audio datasets? `audioclass` provides efficient batch
processing through specialized iterators. These iterators handle the loading,
preprocessing, and batching of your audio recordings, allowing you to focus on
analysis.

### `SimpleIterator`: Your Starting Point

The [`SimpleIterator`][audioclass.batch.SimpleIterator] is a great option for getting started with batch processing:

```python
from audioclass.models.birdnet import BirdNET
from audioclass.batch import SimpleIterator
from soundevent import data

model = BirdNET.load()
recordings = ...  # Load a list of soundevent Recording objects
iterable = SimpleIterator(
    recordings,
    samplerate=model.samplerate,
    input_samples=model.input_samples
)
predictions = model.process_iterable(iterable)
```

### More Flexibility with Data Sources

[`SimpleIterator`][audioclass.batch.SimpleIterator] can handle more than just lists of [`Recording`][soundevent.data.Recording] objects. You can easily create an iterator from:

- **A list of file paths:**

  ```python
  list_of_files = ["path/to/file1.wav", "path/to/file2.wav", "path/to/file3.wav"]
  iterable = SimpleIterator.from_files(
      list_of_files,
      samplerate=model.samplerate,
      input_samples=model.input_samples,
  )
  ```

- **An audio directory:** (Recursively searches for audio files)

  ```python
  audio_directory = "path/to/directory"
  iterable = SimpleIterator.from_directory(
      audio_directory,
      samplerate=model.samplerate,
      input_samples=model.input_samples
  )
  ```

- **A pandas DataFrame:**

  ```python
  import pandas as pd

  df = pd.DataFrame({"path": list_of_files, "latitude": [0, 1, 2], "longitude": [3, 4, 5]})  # Example DataFrame
  iterable = SimpleIterator.from_dataframe(
      df,
      samplerate=model.samplerate,
      input_samples=model.input_samples
  )
  ```

**Alternative Iterators:**

While [`SimpleIterator`][audioclass.batch.SimpleIterator] is convenient, audioclass offers other iterators tailored for specific needs:

- [`TFDatasetIterator`][audioclass.batch.tensorflow.TFDatasetIterator]: Leverages TensorFlow Datasets for enhanced performance on large datasets and parallel processing.

For further details and usage examples, consult the **reference documentation**.

## What's Next?

To unlock the full potential of audioclass, dive into its reference documentation and explore:

- **Base Interfaces:** The
  [`ClipClassificationModel`][audioclass.models.base.ClipClassificationModel]
  class and [`ModelOutput`][audioclass.models.base.ModelOutput] structure provide
  the foundation for building custom models and working with predictions.
- **Preprocessing and Postprocessing:** Learn how to preprocess your audio data
  and convert model outputs into formats suitable for your analysis.
- **Extending audioclass:** Discover how to create your own custom models or
  batch iterators to fit your specific audio classification needs.
