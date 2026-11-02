"""Validate images, remove byte-identical duplicates, and freeze dataset splits."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
SPLITS = {"train", "validation", "test"}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def read_image(path: Path, size: int):
    """One RGB, EXIF-aware resize path shared by training and inference."""
    import numpy as np

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype="float32")


def prepare(root: Path, output: Path, seed: int = 42, audit_path: Path | None = None) -> dict:
    root = root.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing manifest: {output}")
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    records, rejected, duplicates = [], [], []
    seen = {}
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Image symlink points outside the dataset")
            with Image.open(path) as image:
                image.verify()
            read_image(path, 32)  # verify() alone does not fully decode the pixels
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            rejected.append({"path": relative, "reason": str(exc)})
            continue
        sha = digest(path)
        label = path.parent.name
        if sha in seen:
            previous = seen[sha]
            if previous["label"] != label:
                raise ValueError(
                    f"Identical image has conflicting labels: {previous['path']}, {relative}"
                )
            duplicates.append({"path": relative, "duplicate_of": previous["path"]})
            continue
        row = {"path": relative, "label": label, "sha256": sha}
        seen[sha] = row
        records.append(row)
    classes = sorted({row["label"] for row in records})
    if len(classes) < 2:
        raise ValueError("At least two classes with valid images are required")
    groups = defaultdict(list)
    for row in records:
        groups[row["label"]].append(row)
    audit_report = None
    if audit_path:
        audit_report = json.loads(audit_path.read_text())
        if audit_report.get("schema_version") != 1 or not audit_report.get("review_complete"):
            raise ValueError("Audit needs a supported schema and completed similarity reviews")
        audited = {row["path"]: row["sha256"] for row in audit_report["records"]}
        current = {row["path"]: row["sha256"] for row in records}
        for duplicate in duplicates:
            current[duplicate["path"]] = current[duplicate["duplicate_of"]]
        if current != audited:
            raise ValueError("Audit does not match current dataset contents")
        group_labels = defaultdict(set)
        for row in records:
            row["group"] = audit_report["groups"][row["path"]]
            group_labels[row["group"]].add(row["label"])
        if any(len(labels) > 1 for labels in group_labels.values()):
            raise ValueError("A similarity/identity group has conflicting labels; review it first")
    rng = random.Random(seed)
    for label in classes:
        rows = groups[label]
        if len(rows) < 3:
            raise ValueError(f"Class {label!r} needs at least 3 unique, valid images")
        if audit_report:
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["group"]].append(row)
            units = list(grouped.values())
            if len(units) < 3:
                raise ValueError(f"Class {label!r} needs at least 3 independent groups")
            rng.shuffle(units)
            units.sort(key=len, reverse=True)
            targets = {
                "train": len(rows) * 0.7,
                "validation": len(rows) * 0.15,
                "test": len(rows) * 0.15,
            }
            assigned = dict.fromkeys(targets, 0)
            for i, unit in enumerate(units):
                split = (
                    list(targets)[i]
                    if i < 3
                    else min(targets, key=lambda name: assigned[name] / targets[name])
                )
                for row in unit:
                    row["split"] = split
                assigned[split] += len(unit)
            continue
        rng.shuffle(rows)
        held_out = max(1, round(len(rows) * 0.15))
        for i, row in enumerate(rows):
            row["split"] = (
                "test" if i < held_out else ("validation" if i < held_out * 2 else "train")
            )
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "split_policy": "per-class 70/15/15; holdout sizes rounded, minimum one each",
        "classes": classes,
        "records": sorted(records, key=lambda row: row["path"]),
        "rejected": rejected,
        "duplicates": duplicates,
    }
    if audit_report:
        manifest.update(
            {
                "schema_version": 2,
                "audit_sha256": fingerprint(audit_report),
                "split_policy": (
                    "per-class group-disjoint approximately 70/15/15; largest groups first"
                ),
                "group_policy": audit_report["policy"],
            }
        )
    validate_manifest(manifest)
    write_json(output, manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported manifest schema")
    classes = manifest["classes"]
    if len(classes) < 2 or classes != sorted(set(classes)):
        raise ValueError("Manifest classes must be unique and sorted")
    paths, hashes, coverage = set(), set(), set()
    group_splits = {}
    for row in manifest["records"]:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest paths must stay inside the dataset root")
        if row["label"] not in classes or row["split"] not in SPLITS:
            raise ValueError("Invalid manifest label or split")
        if row["path"] in paths or row["sha256"] in hashes:
            raise ValueError("Duplicate path or image hash in manifest")
        if manifest["schema_version"] == 2:
            group = row.get("group")
            if not isinstance(group, str) or not group:
                raise ValueError("Grouped manifests require a group for every image")
            if group in group_splits and group_splits[group] != row["split"]:
                raise ValueError("An image group crosses dataset splits")
            group_splits[group] = row["split"]
        paths.add(row["path"])
        hashes.add(row["sha256"])
        coverage.add((row["label"], row["split"]))
    if coverage != {(label, split) for label in classes for split in SPLITS}:
        raise ValueError("Every class must occur in all three splits")


def load_manifest(path: Path, root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    root = root.resolve()
    for row in manifest["records"]:
        image = (root / row["path"]).resolve()
        if not image.is_relative_to(root) or digest(image) != row["sha256"]:
            raise ValueError(f"Dataset changed or path escaped root: {row['path']}")
    return manifest


def summary(manifest: dict) -> dict:
    return {
        "images": len(manifest["records"]),
        "classes": len(manifest["classes"]),
        "splits": dict(Counter(row["split"] for row in manifest["records"])),
        "class_counts": dict(Counter(row["label"] for row in manifest["records"])),
        "rejected": len(manifest["rejected"]),
        "duplicates_removed": len(manifest["duplicates"]),
    }
