import json

import pytest

from cat_vision.data import digest, fingerprint, write_json
from cat_vision.selection import select_runs


def candidate(root, name, f1, loss, parameters=10):
    run = root / name
    run.mkdir()
    manifest = {"classes": ["A", "B"], "records": []}
    config = {"seed": 42}
    write_json(run / "manifest.json", manifest)
    write_json(run / "config.json", config)
    write_json(run / "classes.json", manifest["classes"])
    (run / "model.keras").write_bytes(f"fake model artifact: {name}".encode())
    write_json(
        run / "provenance.json",
        dict(
            status="complete",
            manifest_sha256=fingerprint(manifest),
            config_sha256=fingerprint(config),
            model_sha256=digest(run / "model.keras"),
            duration_seconds=1,
            trainable_parameters=parameters,
            best_epoch=1,
        ),
    )
    write_json(
        run / "validation_metrics.json",
        dict(
            split="validation",
            classes=manifest["classes"],
            macro_f1=f1,
            accuracy=f1,
            log_loss=loss,
            model_sha256=digest(run / "model.keras"),
            config_sha256=fingerprint(config),
            manifest_sha256=fingerprint(manifest),
        ),
    )
    return run


def test_selection_uses_validation_and_deterministic_tiebreaks(tmp_path):
    first = candidate(tmp_path, "baseline", 0.8, 0.5)
    second = candidate(tmp_path, "fine-tuning", 0.85, 0.6, parameters=100)
    # No test metrics/predictions exist: selection must not read them.
    path = tmp_path / "selection.json"
    result = select_runs([first, second], path)
    assert result["selected_recipe"] == "fine-tuning"
    assert result["test_used_for_selection"] is False
    assert all(row["validation_metrics_sha256"] for row in result["candidates"])
    with pytest.raises(ValueError, match="overwrite"):
        select_runs([first, second], path)
    third = candidate(tmp_path, "lower-loss", 0.85, 0.4, parameters=100)
    fourth = candidate(tmp_path, "simpler", 0.85, 0.4, parameters=10)
    assert select_runs([second, third], tmp_path / "loss.json")["selected_recipe"] == "lower-loss"
    assert select_runs([third, fourth], tmp_path / "simple.json")["selected_recipe"] == "simpler"


def test_selection_rejects_stale_or_incomparable_candidates(tmp_path):
    first = candidate(tmp_path, "first", 0.8, 0.5)
    second = candidate(tmp_path, "second", 0.7, 0.6)
    provenance = json.loads((second / "provenance.json").read_text())
    write_json(second / "manifest.json", {"classes": ["A", "B"], "records": ["different split"]})
    with pytest.raises(ValueError, match="fingerprint"):
        select_runs([first, second], tmp_path / "bad.json")
    provenance["manifest_sha256"] = fingerprint(json.loads((second / "manifest.json").read_text()))
    write_json(second / "provenance.json", provenance)
    metrics = json.loads((second / "validation_metrics.json").read_text())
    metrics["manifest_sha256"] = provenance["manifest_sha256"]
    write_json(second / "validation_metrics.json", metrics)
    with pytest.raises(ValueError, match="same exact"):
        select_runs([first, second], tmp_path / "mixed.json")
    with pytest.raises(ValueError, match="duplicate"):
        select_runs([first, first], tmp_path / "duplicate.json")


@pytest.mark.parametrize("key", ["model_sha256", "config_sha256", "manifest_sha256"])
def test_selection_rejects_foreign_or_legacy_metric_fingerprints(tmp_path, key):
    run = candidate(tmp_path, "candidate", 0.8, 0.5)
    path = run / "validation_metrics.json"
    metrics = json.loads(path.read_text())
    original = dict(metrics)
    metrics[key] = "foreign artifact"
    write_json(path, metrics)
    with pytest.raises(ValueError, match="Validation metrics fingerprints"):
        select_runs([run], tmp_path / "foreign.json")
    original.pop(key)
    write_json(path, original)
    with pytest.raises(ValueError, match="Validation metrics fingerprints"):
        select_runs([run], tmp_path / "legacy.json")


def test_selection_rejects_copied_scores_and_changed_saved_classes(tmp_path):
    first = candidate(tmp_path, "first", 0.8, 0.5)
    second = candidate(tmp_path, "second", 0.99, 0.01)
    original = (first / "validation_metrics.json").read_text()
    (first / "validation_metrics.json").write_text((second / "validation_metrics.json").read_text())
    with pytest.raises(ValueError, match="Validation metrics fingerprints"):
        select_runs([first], tmp_path / "copied.json")
    (first / "validation_metrics.json").write_text(original)
    write_json(first / "classes.json", ["B", "A"])
    with pytest.raises(ValueError, match="class mapping"):
        select_runs([first], tmp_path / "classes.json")
