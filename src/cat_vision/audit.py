"""Find duplicate-image candidates and freeze reviewed grouping evidence before splitting."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from cat_vision.data import EXTENSIONS, digest, write_json


def source_identity(path: str) -> tuple[str, str | None]:
    name = Path(path).name
    match = re.search(r"(?<!-)-(\d+)_", name)
    if match:
        return "petfinder", match.group(1)
    return ("google" if "--" in name else "oxford"), None


def image_signatures(path: Path) -> dict:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = np.asarray(image)
        gray = image.convert("L")
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        spectrum = dctn(small, type=2, norm="ortho")[:8, :8]
        difference = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        phash = sum(int(bit) << i for i, bit in enumerate((spectrum > np.median(spectrum)).flat))
        dhash = sum(
            int(bit) << i for i, bit in enumerate((difference[:, 1:] > difference[:, :-1]).flat)
        )
        return {
            "dimensions": list(image.size),
            "pixel_sha256": hashlib.sha256(str(image.size).encode() + pixels.tobytes()).hexdigest(),
            "phash": f"{phash:016x}",
            "dhash": f"{dhash:016x}",
        }


def build_groups(records: list[dict], pairs: list[dict]) -> dict[str, str]:
    parents = {row["path"]: row["path"] for row in records}

    def find(path):
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    def join(left, right):
        parents[find(right)] = find(left)

    seen_pixels, seen_listings = {}, {}
    for row in records:
        path = row["path"]
        key = row["pixel_sha256"]
        if key in seen_pixels:
            join(path, seen_pixels[key])
        seen_pixels[key] = path
        listing = row["listing_id"]
        if listing:
            if listing in seen_listings:
                join(path, seen_listings[listing])
            seen_listings[listing] = path
    for pair in pairs:
        if pair.get("same_photo") is True:
            join(pair["left"], pair["right"])
    members = defaultdict(list)
    for path in parents:
        members[find(path)].append(path)
    groups = {}
    for paths in members.values():
        identifier = "g_" + hashlib.sha256("\0".join(sorted(paths)).encode()).hexdigest()[:16]
        groups.update(dict.fromkeys(paths, identifier))
    return groups


def audit(root: Path, output: Path, decisions: Path | None = None) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing audit: {output}")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    records, rejected = [], []
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Image symlink points outside the dataset")
            signatures = image_signatures(path)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            rejected.append({"path": relative, "reason": str(exc)})
            continue
        source, listing = source_identity(relative)
        records.append(
            {
                "path": relative,
                "label": path.parent.name,
                "sha256": digest(path),
                "source": source,
                "listing_id": listing,
                **signatures,
            }
        )
    if not records:
        raise ValueError("No valid images to audit")
    supplied = json.loads(decisions.read_text()) if decisions else []
    reviews = {}
    for decision in supplied:
        if type(decision.get("same_photo")) is not bool:
            raise ValueError("Each review decision needs a boolean same_photo")
        key = tuple(sorted((decision["left"], decision["right"])))
        if key in reviews:
            raise ValueError("Duplicate review decision")
        reviews[key] = decision
    candidates = []
    phashes = [int(row["phash"], 16) for row in records]
    dhashes = [int(row["dhash"], 16) for row in records]
    for i, right in enumerate(records):
        for j in range(i):
            left = records[j]
            if left["sha256"] == right["sha256"]:
                continue
            pd = (phashes[i] ^ phashes[j]).bit_count()
            dd = (dhashes[i] ^ dhashes[j]).bit_count()
            if pd > 6 and dd > 4:
                continue
            key = tuple(sorted((left["path"], right["path"])))
            review = reviews.pop(key, {})
            candidates.append(
                {
                    "left": left["path"],
                    "right": right["path"],
                    "phash_distance": pd,
                    "dhash_distance": dd,
                    "same_photo": review.get("same_photo"),
                    "note": review.get("note", ""),
                }
            )
    if reviews:
        raise ValueError("Review decisions contain pairs outside this dataset/candidate policy")
    groups = build_groups(records, candidates)
    counts = Counter(row["sha256"] for row in records)
    group_sizes = Counter(groups.values())
    result = {
        "schema_version": 1,
        "policy": (
            "RGB pixel equality + filename PetFinder IDs + reviewed pHash<=6 OR dHash<=4 pairs"
        ),
        "review_complete": all(pair["same_photo"] is not None for pair in candidates),
        "records": records,
        "rejected": rejected,
        "similar_pairs": candidates,
        "groups": groups,
        "summary": {
            "valid_images": len(records),
            "rejected_images": len(rejected),
            "exact_duplicate_groups": sum(value > 1 for value in counts.values()),
            "redundant_exact_files": sum(value - 1 for value in counts.values()),
            "candidate_pairs": len(candidates),
            "approved_pairs": sum(pair["same_photo"] is True for pair in candidates),
            "groups": len(group_sizes),
            "multi_image_groups": sum(value > 1 for value in group_sizes.values()),
            "source_counts": dict(Counter(row["source"] for row in records)),
        },
    }
    write_json(output, result)
    return result
