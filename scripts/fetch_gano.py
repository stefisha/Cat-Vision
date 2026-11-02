"""Fetch the public, checksum-pinned Gano v3 archive without Kaggle credentials."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://www.kaggle.com/api/v1/datasets/download/shawngano/"
    "gano-cat-breed-image-collection?datasetVersionNumber=3"
)
SHA256 = "4af0320b13d0dff4cf2ab332557d7d0eba60624115779e854968dad25ba40c15"
FOLDER = "Gano-Cat-Breeds-V1_1"


def fetch(archive: Path, destination: Path) -> Path:
    target = destination / FOLDER
    if target.exists():
        raise ValueError(f"Refusing to overwrite extracted data: {target}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(URL, timeout=60) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
        partial.replace(archive)
    with archive.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != SHA256:
        raise ValueError(f"Archive checksum mismatch: {actual}")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not (staging / name).resolve().is_relative_to(staging):
                    raise ValueError("Unsafe archive path")
            zipped.extractall(staging)
        (staging / FOLDER).rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/source/gano-v3.zip"))
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(fetch(args.archive, args.destination))
