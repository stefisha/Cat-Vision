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


@pytest.mark.parametrize("command", ["audit", "prepare", "train", "evaluate", "select", "report"])
def test_new_cli_options_dispatch_from_another_directory(tmp_path, monkeypatch, capsys, command):
    import importlib

    from cat_vision import cli

    data, manifest, run = [tmp_path / name for name in ("cats", "manifest.json", "run")]
    output, evidence, parent = [tmp_path / name for name in ("out.json", "audit.json", "parent")]
    config = tmp_path / "config.json"
    config.write_text('{"seed": 42}')
    mock = Mock()
    if command == "audit":
        monkeypatch.setattr(importlib.import_module("cat_vision.audit"), "audit", mock)
        mock.return_value = {"summary": {"ok": True}}
        flags = ["--data-dir", data, "--output", output, "--decisions", evidence]
        expected = ((data, output, evidence), {})
    elif command == "prepare":
        monkeypatch.setattr(cli, "prepare", mock)
        monkeypatch.setattr(cli, "summary", Mock(return_value={"ok": True}))
        flags = ["--data-dir", data, "--manifest", manifest, "--audit", evidence, "--seed", "9"]
        expected = ((data, manifest, 9, evidence), {})
    elif command == "train":
        monkeypatch.setattr(importlib.import_module("cat_vision.pipeline"), "train", mock)
        flags = [
            "--data-dir",
            data,
            "--manifest",
            manifest,
            "--run-dir",
            run,
            "--config",
            config,
            "--initial-run",
            parent,
        ]
        expected = ((data, manifest, run, {"seed": 42}), {"initial_run": parent})
    elif command == "evaluate":
        monkeypatch.setattr(importlib.import_module("cat_vision.pipeline"), "evaluate", mock)
        mock.return_value = {"ok": True}
        flags = ["--data-dir", data, "--run-dir", run, "--split", "validation"]
        expected = ((data, run, "validation"), {})
    elif command == "select":
        monkeypatch.setattr(importlib.import_module("cat_vision.selection"), "select_runs", mock)
        mock.return_value = {"ok": True}
        flags = ["--run-dir", run, "--run-dir", parent, "--output", output]
        expected = (([run, parent], output), {})
    else:
        monkeypatch.setattr(importlib.import_module("cat_vision.report"), "make_report", mock)
        mock.return_value = output
        flags = [
            "--run-dir",
            run,
            "--output-dir",
            output,
            "--data-dir",
            data,
            "--include-images",
            "--split",
            "validation",
        ]
        expected = ((run, output, data, True, "validation"), {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["cat-vision", command, *map(str, flags)])
    main()
    mock.assert_called_once_with(*expected[0], **expected[1])
    assert json.loads(capsys.readouterr().out)
