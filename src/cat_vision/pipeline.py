"""Train, evaluate and predict with explicit artifact and data contracts."""

import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from cat_vision.data import digest, fingerprint, load_manifest, read_image, write_json
from cat_vision.model import build_model


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('Install ML dependencies with: pip install -e ".[ml]"') from exc
    return tf


def dataset(manifest: dict, root: Path, split: str, size: int, batch_size: int, seed=42):
    tf = tensorflow()
    rows = [row for row in manifest["records"] if row["split"] == split]
    mapping = {label: i for i, label in enumerate(manifest["classes"])}

    paths = [str(root / row["path"]) for row in rows]
    labels = np.array([mapping[row["label"]] for row in rows], dtype=np.int32)
    result = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        # Shuffle paths before decoding so the buffer does not hold full-resolution tensors.
        result = result.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        pixels = tf.numpy_function(
            lambda value: read_image(Path(value.decode("utf-8")), size), [path], tf.float32
        )
        pixels.set_shape((size, size, 3))
        return pixels, label

    return result.map(decode, num_parallel_calls=1).batch(batch_size).prefetch(1)


def train(root: Path, manifest_path: Path, run: Path, config: dict) -> None:
    expected = {
        "seed",
        "image_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "patience",
        "weights",
    }
    if set(config) != expected:
        raise ValueError(f"Config must contain exactly: {sorted(expected)}")
    for key in ("seed", "image_size", "batch_size", "epochs", "patience"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    if (
        config["image_size"] < 32
        or not isinstance(config["learning_rate"], int | float)
        or isinstance(config["learning_rate"], bool)
        or not np.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
    ):
        raise ValueError("Image size must be >=32 and learning rate positive")
    if config["weights"] not in ("imagenet", None):
        raise ValueError("weights must be 'imagenet' or null (smoke testing only)")
    manifest = load_manifest(manifest_path, root)
    if run.exists():
        raise ValueError(f"Run directory already exists: {run}")
    # Capture the source state before this run creates any output files.
    source_root = Path(__file__).resolve().parents[2]
    try:
        if not (source_root / "pyproject.toml").is_file():
            raise FileNotFoundError("Source checkout unavailable")
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    tf = tensorflow()
    started = time.perf_counter()
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    run.mkdir(parents=True)
    write_json(run / "config.json", config)
    write_json(run / "manifest.json", manifest)
    write_json(run / "classes.json", manifest["classes"])
    write_json(
        run / "provenance.json",
        {
            "manifest_sha256": fingerprint(manifest),
            "config_sha256": fingerprint(config),
            "git_revision": revision,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "created_at": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "devices": [device.name for device in tf.config.list_physical_devices()],
            "packages": {
                dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            },
        },
    )
    model = build_model(len(manifest["classes"]), config["image_size"], config["weights"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    args = (manifest, root)
    kwargs = {
        "size": config["image_size"],
        "batch_size": config["batch_size"],
        "seed": config["seed"],
    }
    history = model.fit(
        dataset(*args, "train", **kwargs),
        validation_data=dataset(*args, "validation", **kwargs),
        epochs=config["epochs"],
        shuffle=False,  # The tf.data training pipeline owns shuffling.
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                run / "model.keras", monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config["patience"], restore_best_weights=True
            ),
            tf.keras.callbacks.CSVLogger(run / "history.csv"),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
    )
    write_json(run / "history.json", history.history)
    if not (run / "model.keras").exists():
        raise RuntimeError("Training did not produce a valid checkpoint")
    if not all(np.all(np.isfinite(values)) for values in history.history.values()):
        raise RuntimeError("Training produced non-finite metrics; do not use this run")
    provenance = json.loads((run / "provenance.json").read_text())
    provenance["model_sha256"] = digest(run / "model.keras")
    provenance["status"] = "complete"
    provenance["duration_seconds"] = time.perf_counter() - started
    provenance["best_epoch"] = int(np.argmin(history.history["val_loss"])) + 1
    write_json(run / "provenance.json", provenance)


def load_run(run: Path):
    tf = tensorflow()
    config = json.loads((run / "config.json").read_text())
    classes = json.loads((run / "classes.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    if provenance.get("status") != "complete":
        raise ValueError("Run did not complete successfully")
    if fingerprint(manifest) != provenance["manifest_sha256"]:
        raise ValueError("Run manifest does not match recorded fingerprint")
    if fingerprint(config) != provenance["config_sha256"]:
        raise ValueError("Run configuration does not match recorded fingerprint")
    if classes != manifest["classes"]:
        raise ValueError("Class mapping differs from the training manifest")
    if digest(run / "model.keras") != provenance["model_sha256"]:
        raise ValueError("Model checkpoint does not match recorded fingerprint")
    model = tf.keras.models.load_model(run / "model.keras")
    if model.output_shape[-1] != len(classes) or model.input_shape[1:3] != (
        config["image_size"],
        config["image_size"],
    ):
        raise ValueError("Model shape does not match run metadata")
    return model, config, classes


def evaluate(root: Path, run: Path) -> dict:
    manifest = load_manifest(run / "manifest.json", root)
    provenance = json.loads((run / "provenance.json").read_text())
    if fingerprint(manifest) != provenance["manifest_sha256"]:
        raise ValueError("Run manifest does not match recorded fingerprint")
    model, config, classes = load_run(run)
    if classes != manifest["classes"]:
        raise ValueError("Class mapping differs from the training manifest")
    rows = [row for row in manifest["records"] if row["split"] == "test"]
    probabilities = model.predict(
        dataset(manifest, root, "test", config["image_size"], config["batch_size"]), verbose=0
    )
    predictions = probabilities.argmax(axis=1)
    targets = np.array([classes.index(row["label"]) for row in rows])
    labels = list(range(len(classes)))
    top = np.argsort(probabilities, axis=1)[:, -min(3, len(classes)) :]
    counts = np.bincount(
        [classes.index(row["label"]) for row in manifest["records"] if row["split"] == "train"],
        minlength=len(classes),
    )
    metrics = {
        "split": "test",
        "samples": len(rows),
        "classes": classes,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "top_3_accuracy": float(np.mean(np.any(top == targets[:, None], axis=1))),
        "majority_baseline_accuracy": float(np.mean(targets == counts.argmax())),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }
    write_json(run / "metrics.json", metrics)
    write_json(
        run / "test_predictions.json",
        [
            {
                "path": row["path"],
                "actual": row["label"],
                "predicted": classes[int(pred)],
                "probabilities": dict(zip(classes, map(float, probs), strict=True)),
            }
            for row, pred, probs in zip(rows, predictions, probabilities, strict=True)
        ],
    )
    return metrics


def predict(run: Path, image: Path, top_k: int = 3) -> list[dict]:
    if top_k < 1:
        raise ValueError("top-k must be positive")
    model, config, classes = load_run(run)
    pixels = read_image(image, config["image_size"])
    probabilities = model.predict(pixels[None], verbose=0)[0]
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [{"breed": classes[int(i)], "score": float(probabilities[i])} for i in indices]
