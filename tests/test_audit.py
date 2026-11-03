import copy
import json

import numpy as np
import pytest
from PIL import Image

from cat_vision.audit import audit, source_identity
from cat_vision.data import digest, prepare, validate_manifest


def test_audit_reviews_and_grouped_split(image_root, tmp_path):
    source = image_root / "Bengal/cat-00.png"
    with Image.open(source) as image:
        pixels = np.array(image)
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels).save(image_root / "Bengal/near-copy.png")
    initial = tmp_path / "initial-audit.json"
    result = audit(image_root, initial)
    assert result["review_complete"] is False
    assert result["summary"]["candidate_pairs"] == 1
    with pytest.raises(ValueError, match="completed similarity"):
        prepare(image_root, tmp_path / "bad-manifest.json", audit_path=initial)
    decisions = tmp_path / "reviews.json"
    pairs = result["similar_pairs"]
    pairs[0]["same_photo"] = True
    for side in ("left", "right"):
        pairs[0][f"{side}_sha256"] = digest(image_root / pairs[0][side])
    decisions.write_text(json.dumps(pairs))
    reviewed = tmp_path / "reviewed-audit.json"
    audit(image_root, reviewed, decisions)
    manifest = prepare(image_root, tmp_path / "manifest.json", audit_path=reviewed)
    assert manifest["schema_version"] == 2
    related = [
        row
        for row in manifest["records"]
        if row["path"] in ("Bengal/cat-00.png", "Bengal/near-copy.png")
    ]
    assert len({row["group"] for row in related}) == 1
    assert len({row["split"] for row in related}) == 1
    assert manifest == prepare(image_root, tmp_path / "repeat.json", audit_path=reviewed)
    changed = copy.deepcopy(manifest)
    related_changed = [row for row in changed["records"] if row["group"] == related[0]["group"]]
    related_changed[0]["split"] = "test" if related_changed[1]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="crosses"):
        validate_manifest(changed)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        prepare(image_root, tmp_path / "changed.json", audit_path=reviewed)


def test_filename_id_grouping_and_sources(image_root, tmp_path):
    (image_root / "Bengal/cat-00.png").rename(image_root / "Bengal/Bengal-123_1.png")
    (image_root / "Bengal/cat-01.png").rename(image_root / "Bengal/Bengal-123_2.png")
    result = audit(image_root, tmp_path / "audit.json")
    assert (
        result["groups"]["Bengal/Bengal-123_1.png"] == result["groups"]["Bengal/Bengal-123_2.png"]
    )
    assert source_identity("Siamese/Siamese-45150207_7156.jpg") == ("petfinder", "45150207")
    assert source_identity("Sphynx/Sphynx--01.jpg") == ("google", None)
    assert source_identity("Birman/Birman_12.jpg") == ("oxford", None)
    assert source_identity("custom/my-cat.jpg") == ("unknown", None)


def test_audit_rejects_stale_or_duplicate_reviews(image_root, tmp_path):
    decisions = tmp_path / "decisions.json"
    entry = {"left": "Bengal/cat-00.png", "right": "Bengal/cat-01.png", "same_photo": True}
    for side in ("left", "right"):
        entry[f"{side}_sha256"] = digest(image_root / entry[side])
    decisions.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="outside this dataset"):
        audit(image_root, tmp_path / "audit.json", decisions)
    decisions.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="Duplicate review"):
        audit(image_root, tmp_path / "audit.json", decisions)


@pytest.mark.parametrize("same_photo", [True, False])
def test_review_rejects_changed_image_even_when_still_similar(image_root, tmp_path, same_photo):
    source = image_root / "Bengal/cat-00.png"
    with Image.open(source) as image:
        pixels = np.array(image)
    pixels[0, 0, 0] ^= 1
    copy = image_root / "Bengal/near-copy.png"
    Image.fromarray(pixels).save(copy)
    result = audit(image_root, tmp_path / "initial.json")
    decision = dict(result["similar_pairs"][0], same_photo=same_photo)
    for side in ("left", "right"):
        decision[f"{side}_sha256"] = digest(image_root / decision[side])
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps([decision]))
    audit(image_root, tmp_path / "valid.json", reviews)
    pixels[0, 1, 0] ^= 1
    Image.fromarray(pixels).save(copy)
    assert audit(image_root, tmp_path / "changed.json")["summary"]["candidate_pairs"] == 1
    with pytest.raises(ValueError, match="Review image hash"):
        audit(image_root, tmp_path / "stale.json", reviews)
    decision.pop("left_sha256")
    reviews.write_text(json.dumps([decision]))
    with pytest.raises(ValueError, match="Review image hash"):
        audit(image_root, tmp_path / "missing-hash.json", reviews)


def test_large_groups_do_not_force_unnecessary_holdout_oversizing(image_root, tmp_path):
    rows = []
    groups = {}
    for label in ("Bengal", "Maine Coon", "Siamese"):
        for i, path in enumerate(sorted((image_root / label).glob("*.png"))):
            relative = path.relative_to(image_root).as_posix()
            rows.append(dict(path=relative, sha256=digest(path)))
            index = 0 if i < 6 else 1 if i < 10 else i - 8
            groups[relative] = f"{label}-{index}"
    report = tmp_path / "audit.json"
    report.write_text(
        json.dumps(
            dict(
                schema_version=1,
                review_complete=True,
                records=rows,
                groups=groups,
                policy="allocation fixture",
            )
        )
    )
    manifest = prepare(image_root, tmp_path / "manifest.json", audit_path=report)
    for label in manifest["classes"]:
        for split in ("validation", "test"):
            assert (
                sum(r["label"] == label and r["split"] == split for r in manifest["records"]) <= 2
            )
    validate_manifest(manifest)
