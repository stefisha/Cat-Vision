"""Command-line entry point; data preparation does not import TensorFlow."""

import argparse
import json
from pathlib import Path

from cat_vision.data import prepare, summary


def main():
    parser = argparse.ArgumentParser(description="Cat Vision · cat breed classification")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Inspect duplicate candidates and identity groups")
    audit_parser.add_argument("--data-dir", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--decisions", type=Path)
    prep = sub.add_parser("prepare", help="Validate images and freeze train/validation/test splits")
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--audit", type=Path, help="Completed audit for group-disjoint splitting")
    train = sub.add_parser("train", help="Train on a frozen manifest")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--initial-run", type=Path, help="Parent run for fine-tuning")
    train.add_argument("--config", type=Path, required=True, help="Path to training settings JSON")
    evaluation = sub.add_parser(
        "evaluate", help="Evaluate the saved checkpoint on validation or test images"
    )
    evaluation.add_argument("--data-dir", type=Path, required=True)
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=["test", "validation"], default="test")
    selection = sub.add_parser("select", help="Freeze a validation-only candidate selection")
    selection.add_argument("--run-dir", type=Path, action="append", required=True)
    selection.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report", help="Render evaluation figures and a local HTML report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--data-dir", type=Path)
    report.add_argument("--include-images", action="store_true", help="Embed local dataset photos")
    report.add_argument("--split", choices=["test", "validation"], default="test")
    prediction = sub.add_parser("predict", help="Return the highest-scoring breeds for one image")
    prediction.add_argument("--run-dir", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            from cat_vision.audit import audit

            result = audit(args.data_dir, args.output, args.decisions)["summary"]
        elif args.command == "select":
            from cat_vision.selection import select_runs

            result = select_runs(args.run_dir, args.output)
        elif args.command == "report":
            from cat_vision.report import make_report

            path = make_report(
                args.run_dir, args.output_dir, args.data_dir, args.include_images, args.split
            )
            result = {"report": str(path)}
        elif args.command == "prepare":
            result = summary(prepare(args.data_dir, args.manifest, args.seed, args.audit))
        else:
            from cat_vision import pipeline

            if args.command == "train":
                kwargs = {"initial_run": args.initial_run} if args.initial_run else {}
                pipeline.train(
                    args.data_dir,
                    args.manifest,
                    args.run_dir,
                    json.loads(args.config.read_text()),
                    **kwargs,
                )
                result = {"run_dir": str(args.run_dir), "status": "trained"}
            elif args.command == "evaluate":
                result = pipeline.evaluate(args.data_dir, args.run_dir, args.split)
            else:
                result = pipeline.predict(args.run_dir, args.image, args.top_k)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
