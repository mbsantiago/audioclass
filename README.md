# audioclass

A Python library that simplifies the process of using machine learning models for audio classification.

## Description

Audioclass provides a unified interface for working with various audio
classification models, making it easier to load, preprocess, batch process, and
analyze audio data. It offers:

- **Standardized Model Interface:** Easily swap between different model implementations (TensorFlow, TensorFlow Lite, etc.) without changing your code.
- **Flexible Data Loading:** Load audio data from files, directories, or pandas DataFrames with a few simple commands.
- **Efficient Batch Processing:** Effortlessly process large datasets in batches for faster inference.
- **Unified Postprocessing:** Convert model outputs into easy-to-use formats like [xarray datasets](https://docs.xarray.dev/en/stable/) or [soundevent](https://mbsantiago.github.io/soundevent/) objects.
- **Pre-trained Models:** Get started quickly with built-in support for popular models like BirdNET and Perch.

## Installation

Get started with audioclass in a snap:

```bash
pip install audioclass
```

**Optional Dependencies:**

- **For BirdNET:** Install additional dependencies using `pip install "audioclass[birdnet]"`.
- **For Perch:** Install additional dependencies using `pip install "audioclass[perch]"`.

## How to Use It

Here's a quick example of how to load the BirdNET model, preprocess an audio file, and get predictions:

```python
from audioclass.models.birdnet import BirdNET

# Load the model
model = BirdNET.load()

# Get predictions
predictions = model.process_file("path/to/audio/file.wav")

print(predictions)
```

For more detailed examples, tutorials, and complete API documentation, visit our [documentation website](https://mbsantiago.github.io/audioclass).

## Contributing

We welcome contributions to audioclass! If you'd like to get involved, please check out our [Contributing Guidelines](CONTRIBUTING.md).

## Attribution

- The **BirdNET** model was developed by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology, in collaboration with Chemnitz University of Technology. This package is not affiliated with the BirdNET project. If you use the BirdNET model, please cite the relevant paper (see the `audioclass.models.birdnet` module docstring for details).

- The **Perch** model is a research product by Google Research. This package is not affiliated with Google Research.

**Audioclass is an independent project and is not affiliated with either the BirdNET or Perch projects.**
