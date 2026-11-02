import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from cat_vision.cli import main


def train_arguments(tmp_path):
    return [
        "cat-vision",
        "train",
        "--data-dir",
        str(tmp_path / "cats"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--run-dir",
        str(tmp_path / "run"),
    ]


def test_train_requires_explicit_config_outside_checkout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", train_arguments(tmp_path))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "required: --config" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_train_loads_absolute_config_outside_checkout(tmp_path, monkeypatch, capsys):
    from cat_vision import pipeline

    config_path = Path(__file__).resolve().parents[1] / "configs/baseline.json"
    train = Mock()
    monkeypatch.setattr(pipeline, "train", train)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [*train_arguments(tmp_path), "--config", str(config_path)])
    main()
    train.assert_called_once_with(
        tmp_path / "cats",
        tmp_path / "manifest.json",
        tmp_path / "run",
        json.loads(config_path.read_text()),
    )
    assert json.loads(capsys.readouterr().out)["status"] == "trained"
