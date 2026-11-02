import copy
import json

import numpy as np
import pytest
from PIL import Image

from cat_vision.audit import audit, source_identity
from cat_vision.data import prepare, validate_manifest


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


def test_audit_rejects_stale_or_duplicate_reviews(image_root, tmp_path):
    decisions = tmp_path / "decisions.json"
    entry = {"left": "Bengal/cat-00.png", "right": "Bengal/cat-01.png", "same_photo": True}
    decisions.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="outside this dataset"):
        audit(image_root, tmp_path / "audit.json", decisions)
    decisions.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="Duplicate review"):
        audit(image_root, tmp_path / "audit.json", decisions)
