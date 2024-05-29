import argparse
from pathlib import Path
from typing import List

import pandas as pd
from audioclass import BirdNET
from audioclass.constants import (
    BATCH_SIZE,
    DEFAULT_THRESHOLD,
    LABELS_PATH,
    MODEL_PATH,
)
from soundevent import data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the BirdNET model file",
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
        type=Path,
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


def main():
    args = parse_args()

    if not args.output.exists():
        args.output.mkdir(parents=True)

    model = BirdNET.from_model_file(
        model_path=args.model,
        labels_path=args.labels,
        confidence_threshold=args.threshold,
    )
    outputs = model.process_directory(
        directory=args.directory,
        recursive=args.recursive,
        batch_size=args.batch_size,
    )
    save_features(outputs, dest=args.output / args.feature_file)
    save_predictions(outputs, dest=args.output / args.prediction_file)


if __name__ == "__main__":
    main()
