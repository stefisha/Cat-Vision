import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest


def load_fetcher():
    path = Path(__file__).resolve().parents[1] / "scripts/fetch_gano.py"
    spec = importlib.util.spec_from_file_location("fetch_gano", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zip_bytes(name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr(name, b"fixture")
    return stream.getvalue()


def test_fetch_verifies_archive_and_refuses_data_overwrite(tmp_path, monkeypatch):
    module = load_fetcher()
    payload = zip_bytes(f"{module.FOLDER}/Bengal/cat.jpg")
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    archive, destination = tmp_path / "archive.zip", tmp_path / "raw"
    result = module.fetch(archive, destination)
    assert (result / "Bengal/cat.jpg").read_bytes() == b"fixture"
    with pytest.raises(ValueError, match="overwrite"):
        module.fetch(archive, destination)


def test_fetch_rejects_changed_archive_and_path_escape(tmp_path, monkeypatch):
    module = load_fetcher()
    archive = tmp_path / "archive.zip"
    payload = zip_bytes("../escape.txt")
    archive.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        module.fetch(archive, tmp_path / "raw")
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="Unsafe"):
        module.fetch(archive, tmp_path / "raw")
    assert not (tmp_path / "escape.txt").exists()


def test_bad_download_never_reaches_archive_and_can_be_retried(tmp_path, monkeypatch):
    module = load_fetcher()
    valid = zip_bytes(f"{module.FOLDER}/Bengal/cat.jpg")
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(valid).hexdigest())
    responses = iter([b"<html>login page with HTTP 200</html>", valid])
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(next(responses))
    )
    archive, destination = tmp_path / "archive.zip", tmp_path / "raw"
    with pytest.raises(ValueError, match="checksum"):
        module.fetch(archive, destination)
    assert not archive.exists()
    assert not archive.with_suffix(".zip.part").exists()
    result = module.fetch(archive, destination)
    assert (result / "Bengal/cat.jpg").exists()
