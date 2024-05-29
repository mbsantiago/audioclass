import argparse
from pathlib import Path
from typing import List

import pandas as pd
from audioclass.batch import BaseIterator, SimpleIterator
from audioclass.batch.tensorflow import TFDatasetIterator
from audioclass.constants import (
    BATCH_SIZE,
    DEFAULT_THRESHOLD,
)
from audioclass.models.birdnet import LABELS_PATH, MODEL_PATH, BirdNET
from soundevent import data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the BirdNET model file",
    )
    parser.add_argument(
        "--iterator-type",
        choices=["simple", "tensorflow"],
        default="simple",
        help="Type of batch iterator to use",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Path to the directory containing audio files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for audio files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for processing audio files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=LABELS_PATH,
        help="Path to the BirdNET labels file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="output",
        help="Directory in which to save predictions",
    )
    parser.add_argument(
        "--prediction_file",
        type=Path,
        default="predictions.parquet",
        help="File in which to save predictions",
    )
    parser.add_argument(
        "--feature_file",
        type=Path,
        default="features.parquet",
        help="File in which to save features",
    )
    return parser.parse_args()


def save_predictions(
    clip_predictions: List[data.ClipPrediction],
    dest: Path,
) -> None:
    data = []
    for clip_prediction in clip_predictions:
        clip = clip_prediction.clip
        recording = clip.recording

        for predicted_tag in clip_prediction.tags:
            data.append(
                {
                    "recording": str(recording.path),
                    "start_time": clip.start_time,
                    "end_time": clip.end_time,
                    "species": predicted_tag.tag.value,
                    "probability": predicted_tag.score,
                }
            )

    pd.DataFrame(data).to_parquet(dest, index=False)


def save_features(
    clip_predictions: List[data.ClipPrediction],
    dest: Path,
):
    data = []
    for clip_prediction in clip_predictions:
        clip = clip_prediction.clip
        recording = clip.recording

        data.append(
            {
                "recording": str(recording.path),
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                **{feat.name: feat.value for feat in clip_prediction.features},
            }
        )

    pd.DataFrame(data).to_parquet(dest, index=False)


def load_iterator(
    iterator_type: str,
    directory: Path,
    samplerate: int,
    input_samples: int,
    batch_size: int,
) -> BaseIterator:
    if iterator_type == "simple":
        return SimpleIterator.from_directory(
            directory=directory,
            samplerate=samplerate,
            input_samples=input_samples,
            batch_size=batch_size,
        )

    if iterator_type == "tensorflow":
        return TFDatasetIterator.from_directory(
            directory=directory,
            samplerate=samplerate,
            input_samples=input_samples,
            batch_size=batch_size,
        )

    raise ValueError(f"Invalid iterator type: {iterator_type}")


def main():
    args = parse_args()

    if not args.output.exists():
        args.output.mkdir(parents=True)

    model = BirdNET.load(
        model_path=args.model,
        labels_path=args.labels,
        confidence_threshold=args.threshold,
    )
    iterator = load_iterator(
        iterator_type=args.iterator_type,
        directory=args.directory,
        samplerate=model.samplerate,
        input_samples=model.input_samples,
        batch_size=args.batch_size,
    )
    outputs = model.process_iterable(iterator)  # type: ignore
    save_features(outputs, dest=args.output / args.feature_file)
    save_predictions(outputs, dest=args.output / args.prediction_file)


if __name__ == "__main__":
    main()
