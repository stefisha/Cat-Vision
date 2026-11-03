"""Render measured evaluation artifacts as figures and a self-contained local HTML report."""

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cat_vision.data import read_image


def make_report(
    run: Path,
    output: Path | None = None,
    data_root: Path | None = None,
    include_images: bool = False,
    split: str = "test",
) -> Path:
    if split not in ("validation", "test"):
        raise ValueError("Report split must be validation or test")
    if include_images and data_root is None:
        raise ValueError("--include-images requires --data-dir")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Install report dependencies with: pip install -e ".[report]"') from exc
    metrics_name = "metrics.json" if split == "test" else "validation_metrics.json"
    metrics = json.loads((run / metrics_name).read_text())
    predictions = json.loads((run / f"{split}_predictions.json").read_text())
    history = json.loads((run / "history.json").read_text())
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    output = output or run / f"{split}-report"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    classes = metrics["classes"]
    matrix = np.asarray(metrics["confusion_matrix"])
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0
    )
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j]:
                axis.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[i, j] > 0.5 else "#18293d",
                )
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{split.title()} confusion matrix · counts; color = recall within each actual label",
    )
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    figure.colorbar(image, ax=axis, label="Fraction of actual-label images")
    figure.savefig(output / "confusion-matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)
    for axis, key, title in zip(
        axes, ["loss", "accuracy"], ["Cross-entropy loss", "Accuracy"], strict=True
    ):
        axis.plot(epochs, history[key], label="Train", color="#276b8c")
        axis.plot(epochs, history[f"val_{key}"], label="Validation", color="#be6a30")
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.legend()
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output / "learning-curves.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    ordered = sorted(classes, key=lambda name: metrics["classification_report"][name]["recall"])
    figure, axis = plt.subplots(figsize=(10, max(4, len(classes) * 0.38)))
    recalls = [metrics["classification_report"][name]["recall"] for name in ordered]
    axis.barh(ordered, recalls, color="#397d8d")
    for i, name in enumerate(ordered):
        support = metrics["classification_report"][name]["support"]
        axis.text(
            recalls[i] + 0.015,
            i,
            f"{recalls[i]:.1%} · n={int(support)}",
            va="center",
            fontsize=9,
        )
    axis.set(xlim=(0, 1), xlabel="Recall", title=f"{split.title()} recall by source-provided label")
    figure.savefig(output / "per-label-recall.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    confidence = np.array([max(row["probabilities"].values()) for row in predictions])
    correct = np.array([row["actual"] == row["predicted"] for row in predictions])
    figure, axis = plt.subplots(figsize=(6, 5))
    bins = np.minimum((confidence * 10).astype(int), 9)
    xs, ys, sizes = [], [], []
    for index in range(10):
        mask = bins == index
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            sizes.append(int(mask.sum()))
    axis.plot([0, 1], [0, 1], "--", color="#87939d", label="Perfect calibration")
    axis.scatter(xs, ys, s=np.sqrt(sizes) * 20, color="#276b8c", label="Observed bins")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean maximum softmax score",
        ylabel="Observed accuracy",
        title="Reliability diagram · uncalibrated scores",
    )
    axis.legend()
    figure.savefig(output / "reliability.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    def escape(value):
        return html.escape(str(value), quote=True)

    rows = []
    for label in classes:
        item = metrics["classification_report"][label]
        rows.append(
            f"<tr><th>{escape(label)}</th><td>{item['precision']:.1%}</td>"
            f"<td>{item['recall']:.1%}</td><td>{item['f1-score']:.3f}</td>"
            f"<td>{int(item['support'])}</td></tr>"
        )
    mistakes = sorted(
        [row for row in predictions if row["actual"] != row["predicted"]],
        key=lambda row: max(row["probabilities"].values()),
        reverse=True,
    )[:12]
    cards = []
    for row in mistakes:
        thumbnail = ""
        if include_images:
            root = data_root.resolve()
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Prediction image path escapes data root")
            buffer = io.BytesIO()
            Image.fromarray(read_image(path, 224).astype("uint8")).save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            thumbnail = f'<img alt="{escape(row["path"])}" src="data:image/jpeg;base64,{encoded}">'
        cards.append(
            f"<article>{thumbnail}<p><strong>{escape(row['actual'])}</strong> → "
            f"{escape(row['predicted'])}</p><p>Score: "
            f"{max(row['probabilities'].values()):.1%}</p><small>{escape(row['path'])}</small></article>"
        )
    uncertainty = metrics.get("uncertainty", {})
    interval = uncertainty.get("accuracy")
    interval_text = (
        f"; 95% group-bootstrap interval {interval[0]:.1%}–{interval[1]:.1%}" if interval else ""
    )
    style = """
body{font-family:system-ui,sans-serif;color:#18293d;background:#f5f6f8;margin:0}
main{max-width:1100px;margin:40px auto;padding:28px;background:white;border-radius:16px}
h1,h2{color:#14273f}p{line-height:1.6}img{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%}
th,td{padding:10px;border-bottom:1px solid #dce2e8;text-align:left}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
article{padding:16px;background:#f2f5f7;border-radius:10px;overflow-wrap:anywhere}
code,pre{overflow-wrap:anywhere;white-space:pre-wrap}
small{color:#506274}figure{margin:20px 0}a{color:#236985}
"""
    figures = ""
    for name, title in [
        ("learning-curves", "Training history"),
        ("confusion-matrix", "Label confusions"),
        ("per-label-recall", "Per-label recall"),
        ("reliability", "Score reliability"),
    ]:
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode("ascii")
        figures += (
            f'<h2>{title}</h2><figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{title}"></figure>'
        )

    def data_table(caption, headers, data):
        heading = "".join(f'<th scope="col">{escape(value)}</th>' for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in data
        )
        return (
            f"<details><summary>{escape(caption)} (data table)</summary>"
            f"<table><caption>{escape(caption)}</caption><thead><tr>{heading}</tr></thead>"
            f"<tbody>{body}</tbody></table></details>"
        )

    accessible_data = data_table(
        "Learning curves",
        ["Epoch", "Train loss", "Validation loss", "Train accuracy", "Validation accuracy"],
        [
            [
                i + 1,
                f"{history['loss'][i]:.4f}",
                f"{history['val_loss'][i]:.4f}",
                f"{history['accuracy'][i]:.1%}",
                f"{history['val_accuracy'][i]:.1%}",
            ]
            for i in range(len(history["loss"]))
        ],
    )
    accessible_data += data_table(
        "Confusion matrix counts (rows: actual, columns: predicted)",
        ["Actual label", *classes],
        [[label, *matrix[i].tolist()] for i, label in enumerate(classes)],
    )
    accessible_data += data_table(
        "Reliability bins",
        ["Mean score", "Accuracy", "Images"],
        [[f"{x:.4f}", f"{y:.1%}", n] for x, y, n in zip(xs, ys, sizes, strict=True)],
    )
    photo_note = (
        "Photographs are embedded in this local report. "
        "Review source-image rights before sharing it."
        if include_images
        else "Dataset photographs are omitted from this report."
    )
    taxonomy_note = "Tuxedo is a coat pattern. " if "Tuxedo" in classes else ""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cat Vision · {escape(split)} evaluation</title><style>{style}</style></head><body><main>
<h1>Cat Vision · {escape(split)} evaluation</h1>
<p>{metrics["samples"]} images · {len(classes)} source-provided labels</p>
<p><strong>Accuracy {metrics["accuracy"]:.1%}</strong>{interval_text}<br>
Macro F1 {metrics["macro_f1"]:.3f} · Top-3 accuracy {metrics["top_3_accuracy"]:.1%} ·
Majority baseline {metrics["majority_baseline_accuracy"]:.1%}</p>
<p>This measures performance within the audited collection. {taxonomy_note}
Softmax scores are uncalibrated; unknown breeds and non-cat images have not been evaluated.</p>
{figures}
<h2>Accessible diagnostic data</h2>{accessible_data}
<h2>Per-label metrics</h2><table><tr><th>Label</th><th>Precision</th><th>Recall</th>
<th>F1</th><th>Support</th></tr>{"".join(rows)}</table>
<h2>Most confident mistakes</h2>
<p>Up to twelve incorrect predictions, ranked by maximum softmax score.
This selection deliberately highlights confident errors;
it is not a representative image sample.</p>
<div class="cards">{"".join(cards)}</div><p>{photo_note}</p>
<h2>Run record</h2><pre>{escape(json.dumps(config, indent=2))}</pre>
<p>Revision: <code>{escape(provenance.get("git_revision"))}</code> ·
Dirty at start: {escape(provenance.get("git_dirty"))}<br>
Manifest SHA-256: <code>{escape(provenance.get("manifest_sha256"))}</code></p>
</main></body></html>"""
    path = output / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
