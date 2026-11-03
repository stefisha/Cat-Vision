import numpy as np

from cat_vision.metrics import uncertainty_intervals


def test_group_bootstrap_is_reproducible_and_keeps_paired_images_together():
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 0])
    rows = [dict(path=str(i), group="a" if i < 2 else "b") for i in range(4)]
    first = uncertainty_intervals(targets, predictions, rows, [0, 1], resamples=100)
    assert first == uncertainty_intervals(targets, predictions, rows, [0, 1], resamples=100)
    # Only one group per stratum: selecting a group always selects both images.
    assert first["accuracy"] == [0.5, 0.5]
    assert first["macro_f1"] == [0.5, 0.5]
