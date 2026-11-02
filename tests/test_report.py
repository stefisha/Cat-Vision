import json

import pytest

from cat_vision.report import make_report


def test_report_generates_figures_and_escapes_labels(tmp_path):
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    run.mkdir()
    classes = ["<Cat>", "Other"]
    report = {
        label: dict(precision=0.5, recall=0.5, **{"f1-score": 0.5}, support=2) for label in classes
    }
    metrics = dict(
        classes=classes,
        samples=4,
        accuracy=0.5,
        macro_f1=0.5,
        top_3_accuracy=1,
        majority_baseline_accuracy=0.5,
        classification_report=report,
        confusion_matrix=[[1, 1], [1, 1]],
    )
    predictions = [
        dict(
            path="breed/cat.png",
            actual="<Cat>",
            predicted="Other",
            probabilities={"<Cat>": 0.1, "Other": 0.9},
        )
    ]
    files = {
        "metrics.json": metrics,
        "test_predictions.json": predictions,
        "history.json": dict(
            loss=[1, 0.5], val_loss=[1, 0.8], accuracy=[0.5, 0.8], val_accuracy=[0.5, 0.6]
        ),
        "config.json": {"augmentation": "none"},
        "provenance.json": {"git_dirty": False},
    }
    for name, value in files.items():
        (run / name).write_text(json.dumps(value))
    path = make_report(run)
    document = path.read_text()
    assert "&lt;Cat&gt;" in document
    assert "<Cat>" not in document
    assert "data:image/jpeg" not in document
    assert document.count("data:image/png") == 4
    assert "Most confident mistakes" in document
    for name in ("confusion-matrix", "learning-curves", "per-label-recall", "reliability"):
        assert (path.parent / f"{name}.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="requires --data-dir"):
        make_report(run, include_images=True)
