"""Measure warm-model decode/resize + inference on evenly spaced validation images."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from cat_vision.data import digest, load_manifest, read_image, write_json
from cat_vision.pipeline import load_run, tensorflow


def benchmark(run: Path, root: Path, samples=50, warmups=5, intra=6, inter=1) -> dict:
    if samples < 1 or warmups < 1 or intra < 1 or inter < 1:
        raise ValueError("Sample, warmup and thread counts must be positive")
    tf = tensorflow()
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
    manifest = load_manifest(run / "manifest.json", root)
    validation = [row for row in manifest["records"] if row["split"] == "validation"]
    if samples > len(validation):
        raise ValueError("Sample count exceeds available validation images")
    started = time.perf_counter()
    model, config, _ = load_run(run)
    load_seconds = time.perf_counter() - started
    first = read_image(root / validation[0]["path"], config["image_size"])[None]
    for _ in range(warmups):
        model(first, training=False).numpy()
    timings = []
    for index in np.linspace(0, len(validation) - 1, samples, dtype=int):
        start = time.perf_counter()
        pixels = read_image(root / validation[index]["path"], config["image_size"])[None]
        model(pixels, training=False).numpy()
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "samples": samples,
        "warmup_calls": warmups,
        "batch_size": 1,
        "includes": "Pillow decode/EXIF/RGB/resize + warm Keras inference",
        "model_sha256": digest(run / "model.keras"),
        "model_bytes": (run / "model.keras").stat().st_size,
        "model_load_seconds": load_seconds,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "tensorflow_intraop_threads": intra,
        "tensorflow_interop_threads": inter,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "scope": "One local CPU session; not a service or cold-start benchmark",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--intra-threads", type=int, default=6)
    parser.add_argument("--inter-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve the previous measurement")
    result = benchmark(
        args.run_dir,
        args.data_dir,
        args.samples,
        args.warmups,
        args.intra_threads,
        args.inter_threads,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
