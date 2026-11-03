import copy
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from cat_vision.data import load_manifest, prepare, read_image, validate_manifest


def test_split_is_reproducible_stratified_and_disjoint(image_root, tmp_path):
    first = prepare(image_root, tmp_path / "first.json")
    second = prepare(image_root, tmp_path / "second.json")
    assert first == second
    assert first["classes"] == ["Bengal", "Maine Coon", "Siamese"]
    hashes = []
    for split, expected in (("train", 24), ("validation", 6), ("test", 6)):
        rows = [row for row in first["records"] if row["split"] == split]
        assert len(rows) == expected
        assert {row["label"] for row in rows} == set(first["classes"])
        hashes.extend(row["sha256"] for row in rows)
    assert len(hashes) == len(set(hashes))
    assert load_manifest(tmp_path / "first.json", image_root) == first
    assert prepare(image_root, tmp_path / "other-seed.json", seed=9)["records"] != first["records"]


def test_bad_images_reported_duplicates_removed(image_root, tmp_path):
    (image_root / "Bengal" / "broken.jpg").write_bytes(b"not an image")
    shutil.copy(image_root / "Bengal/cat-00.png", image_root / "Bengal/copy.png")
    manifest = prepare(image_root, tmp_path / "manifest.json")
    assert len(manifest["records"]) == 36
    assert len(manifest["rejected"]) == len(manifest["duplicates"]) == 1


def test_conflicting_duplicate_labels_fail(image_root, tmp_path):
    shutil.copy(image_root / "Bengal/cat-00.png", image_root / "Siamese/copy.png")
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare(image_root, tmp_path / "manifest.json")


def test_changed_data_rejected(image_root, tmp_path):
    path = tmp_path / "manifest.json"
    prepare(image_root, path)
    (image_root / "Bengal/cat-00.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        load_manifest(path, image_root)


def test_manifest_not_overwritten(image_root, tmp_path):
    path = tmp_path / "manifest.json"
    prepare(image_root, path)
    with pytest.raises(ValueError, match="overwrite"):
        prepare(image_root, path)


@pytest.mark.parametrize("mutation", ["traversal", "duplicate", "missing_split", "classes"])
def test_invalid_manifest_rejected(image_root, tmp_path, mutation):
    manifest = copy.deepcopy(prepare(image_root, tmp_path / "manifest.json"))
    if mutation == "traversal":
        manifest["records"][0]["path"] = "../outside.jpg"
    elif mutation == "duplicate":
        manifest["records"].append(manifest["records"][0])
    elif mutation == "missing_split":
        for row in manifest["records"]:
            row["split"] = "train"
    else:
        manifest["classes"].reverse()
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_minimum_samples_and_empty_data(tmp_path):
    root = tmp_path / "tiny"
    root.mkdir()
    with pytest.raises(ValueError, match="At least two"):
        prepare(root, tmp_path / "empty.json")
    for i, label in enumerate(("A", "B")):
        (root / label).mkdir()
        Image.new("RGB", (8, 8), (i, 0, 0)).save(root / label / "cat.png")
    with pytest.raises(ValueError, match="at least 3"):
        prepare(root, tmp_path / "tiny.json")


def test_image_loader_rgb_shape_and_pixel_scale(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (8, 12), 128).save(path)
    pixels = read_image(path, 32)
    assert pixels.shape == (32, 32, 3)
    assert pixels.dtype == np.float32
    assert np.all(pixels == 128)


def test_cli_prepare(image_root, tmp_path, capsys):
    import sys
    from unittest.mock import patch

    from cat_vision.cli import main

    with patch.object(
        sys,
        "argv",
        [
            "cat-vision",
            "prepare",
            "--data-dir",
            str(image_root),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    ):
        main()
    assert json.loads(capsys.readouterr().out)["images"] == 36


def test_outside_symlink_is_reported(image_root, tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), "red").save(outside)
    (image_root / "Bengal/link.png").symlink_to(outside)
    manifest = prepare(image_root, tmp_path / "manifest.json")
    assert len(manifest["records"]) == 36
    assert "outside" in manifest["rejected"][0]["reason"]


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": "0.1"},
        {"batch_size": 0},
        {"weights": "unknown"},
        {"unexpected": 1},
    ],
)
def test_invalid_config_fails_before_data_access(tmp_path, change):
    from cat_vision.pipeline import train

    config = {
        "seed": 42,
        "image_size": 32,
        "batch_size": 4,
        "epochs": 2,
        "learning_rate": 0.001,
        "patience": 1,
        "weights": None,
    }
    config.update(change)
    with pytest.raises(ValueError):
        train(
            tmp_path / "missing-data", tmp_path / "missing-manifest.json", tmp_path / "run", config
        )
    assert not (tmp_path / "run").exists()


def test_oversized_image_is_rejected_without_aborting(image_root, tmp_path, monkeypatch):
    Image.new("RGB", (80, 80), "red").save(image_root / "Bengal/oversized.png")
    # Valid fixtures have 1,920 pixels; 6,400 exceeds twice this reduced safety limit.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2500)
    path = tmp_path / "manifest.json"
    manifest = prepare(image_root, path)
    assert len(manifest["records"]) == 36
    assert len(manifest["rejected"]) == 1
    assert manifest["rejected"][0]["path"] == "Bengal/oversized.png"
    assert "decompression bomb" in manifest["rejected"][0]["reason"]
    assert load_manifest(path, image_root) == manifest
    assert Image.MAX_IMAGE_PIXELS == 2500


def test_config_error_names_required_optional_and_unknown_keys(tmp_path):
    from cat_vision.pipeline import train

    config = dict(
        seed=42,
        image_size=32,
        batch_size=4,
        epochs=1,
        learning_rate=0.001,
        patience=1,
        weights=None,
        augmentation="none",
        fine_tune_layer=20,
    )
    with pytest.raises(ValueError) as error:
        train(tmp_path, tmp_path / "absent.json", tmp_path / "run", config)
    message = str(error.value)
    assert "required keys" in message and "optional keys" in message
    assert "fine_tune_layers" in message and "augmentation" in message
    assert "unknown keys: ['fine_tune_layer']" in message
