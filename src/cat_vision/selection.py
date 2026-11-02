"""Choose a completed candidate from validation evidence without reading test predictions."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cat_vision.data import digest, fingerprint, write_json


def select_runs(runs: list[Path], output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing selection: {output}")
    if not runs or len({path.resolve() for path in runs}) != len(runs):
        raise ValueError("Provide at least one candidate, with no duplicate run directories")
    candidates = []
    manifest_hashes = set()
    for run in runs:
        provenance = json.loads((run / "provenance.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        metrics = json.loads((run / "validation_metrics.json").read_text())
        if provenance.get("status") != "complete":
            raise ValueError(f"Incomplete candidate: {run}")
        if (
            fingerprint(manifest) != provenance["manifest_sha256"]
            or fingerprint(config) != provenance["config_sha256"]
            or digest(run / "model.keras") != provenance["model_sha256"]
        ):
            raise ValueError(f"Candidate artifact fingerprint mismatch: {run}")
        if metrics["split"] != "validation" or metrics["classes"] != manifest["classes"]:
            raise ValueError("Candidate scores must use validation and match the class mapping")
        scores = [metrics[key] for key in ("macro_f1", "accuracy", "log_loss")]
        if not np.all(np.isfinite(scores)):
            raise ValueError("Candidate metrics must be finite")
        manifest_hashes.add(provenance["manifest_sha256"])
        candidates.append(
            {
                "recipe": run.name,
                "validation_macro_f1": metrics["macro_f1"],
                "validation_accuracy": metrics["accuracy"],
                "validation_log_loss": metrics["log_loss"],
                "checkpoint_sha256": provenance["model_sha256"],
                "validation_metrics_sha256": digest(run / "validation_metrics.json"),
                "training_seconds": provenance["duration_seconds"],
                "trainable_parameters": provenance["trainable_parameters"],
                "best_epoch": provenance["best_epoch"],
            }
        )
    if len(manifest_hashes) != 1:
        raise ValueError("Candidates must use the same exact dataset manifest")
    if len({candidate["recipe"] for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate run names must be unique")
    selected = max(
        candidates,
        key=lambda row: (
            row["validation_macro_f1"],
            -row["validation_log_loss"],
            -row["trainable_parameters"],
        ),
    )
    result = {
        "selected_at": datetime.now(UTC).isoformat(),
        "criterion": (
            "largest validation macro F1; ties: lower log loss, then fewer trainable parameters"
        ),
        "manifest_sha256": next(iter(manifest_hashes)),
        "selected_recipe": selected["recipe"],
        "candidates": candidates,
        "test_used_for_selection": False,
    }
    write_json(output, result)
    return result
