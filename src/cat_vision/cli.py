"""Command-line entry point; data preparation does not import TensorFlow."""

import argparse
import json
from pathlib import Path

from cat_vision.data import prepare, summary


def main():
    parser = argparse.ArgumentParser(description="Cat Vision · cat breed classification")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Validate images and freeze train/validation/test splits")
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Train on a frozen manifest")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    evaluation = sub.add_parser("evaluate", help="Evaluate the saved checkpoint on the test split")
    evaluation.add_argument("--data-dir", type=Path, required=True)
    evaluation.add_argument("--run-dir", type=Path, required=True)
    prediction = sub.add_parser("predict", help="Return the highest-scoring breeds for one image")
    prediction.add_argument("--run-dir", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = summary(prepare(args.data_dir, args.manifest, args.seed))
        else:
            from cat_vision import pipeline

            if args.command == "train":
                pipeline.train(
                    args.data_dir, args.manifest, args.run_dir, json.loads(args.config.read_text())
                )
                result = {"run_dir": str(args.run_dir), "status": "trained"}
            elif args.command == "evaluate":
                result = pipeline.evaluate(args.data_dir, args.run_dir)
            else:
                result = pipeline.predict(args.run_dir, args.image, args.top_k)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
