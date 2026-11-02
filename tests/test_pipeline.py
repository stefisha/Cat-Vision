import json

import numpy as np
import pytest

from cat_vision.data import prepare

pytestmark = pytest.mark.ml


def test_full_run_roundtrip_and_metadata_guards(image_root, tmp_path):
    tf = pytest.importorskip("tensorflow")
    from cat_vision.pipeline import evaluate, load_run, predict, train

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    manifest = tmp_path / "manifest.json"
    prepare(image_root, manifest)
    run = tmp_path / "run"
    config = {
        "seed": 42,
        "image_size": 32,
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 0.001,
        "patience": 1,
        "weights": None,
    }
    train(image_root, manifest, run, config)
    metrics = evaluate(image_root, run)
    assert metrics["samples"] == 6
    assert np.asarray(metrics["confusion_matrix"]).sum() == 6
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert metrics["top_3_accuracy"] == 1  # all 3 toy labels are in the top 3
    assert metrics["majority_baseline_accuracy"] == pytest.approx(1 / 3)
    result = predict(run, image_root / "Siamese/cat-00.png", 10)
    assert {item["breed"] for item in result} == {"Bengal", "Maine Coon", "Siamese"}
    assert sum(item["score"] for item in result) == pytest.approx(1)
    assert result == predict(run, image_root / "Siamese/cat-00.png", 10)
    model, _, classes = load_run(run)
    scaled = model.get_layer("mobilenet_preprocessing")(tf.constant([0.0, 127.5, 255.0]))
    np.testing.assert_allclose(scaled, [-1, 0, 1], atol=1e-6)
    assert classes == ["Bengal", "Maine Coon", "Siamese"]
    assert len(json.loads((run / "history.json").read_text())["loss"]) == 2
    assert (run / "history.csv").exists()
    assert (run / "provenance.json").exists()
    assert len(json.loads((run / "test_predictions.json").read_text())) == 6
    with pytest.raises(ValueError, match="already exists"):
        train(image_root, manifest, run, config)
    with pytest.raises(ValueError, match="positive"):
        predict(run, image_root / "Siamese/cat-00.png", 0)
    (run / "classes.json").write_text(json.dumps(classes[::-1]))
    with pytest.raises(ValueError, match="Class mapping"):
        evaluate(image_root, run)
    with pytest.raises(ValueError, match="Class mapping"):
        predict(run, image_root / "Siamese/cat-00.png")
    (run / "classes.json").write_text(json.dumps(classes))
    changed_config = dict(config, image_size=64)
    (run / "config.json").write_text(json.dumps(changed_config))
    with pytest.raises(ValueError, match="configuration"):
        load_run(run)
    (run / "config.json").write_text(json.dumps(config))
    provenance = json.loads((run / "provenance.json").read_text())
    changed_provenance = dict(provenance, model_sha256="incorrect")
    (run / "provenance.json").write_text(json.dumps(changed_provenance))
    with pytest.raises(ValueError, match="checkpoint"):
        load_run(run)
    (run / "provenance.json").write_text(json.dumps(dict(provenance, status="incomplete")))
    with pytest.raises(ValueError, match="complete"):
        load_run(run)
    (run / "provenance.json").write_text(json.dumps(provenance))
    saved_manifest = json.loads((run / "manifest.json").read_text())
    saved_manifest["seed"] = 100
    (run / "manifest.json").write_text(json.dumps(saved_manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate(image_root, run)
