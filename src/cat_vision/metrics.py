"""Sampling uncertainty for held-out scores, preserving audited image groups."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


def uncertainty_intervals(targets, predictions, rows, labels, resamples=2000, seed=42):
    strata = defaultdict(lambda: defaultdict(list))
    for i, (target, row) in enumerate(zip(targets, rows, strict=True)):
        strata[int(target)][row.get("group", row["path"])].append(i)
    rng = np.random.default_rng(seed)
    accuracy, macro_f1 = [], []
    for _ in range(resamples):
        indices = []
        for groups in strata.values():
            units = list(groups.values())
            for selected in rng.integers(0, len(units), len(units)):
                indices.extend(units[selected])
        actual, guessed = targets[indices], predictions[indices]
        accuracy.append(float(np.mean(actual == guessed)))
        macro_f1.append(f1_score(actual, guessed, labels=labels, average="macro", zero_division=0))
    return {
        "method": "class-stratified group bootstrap; percentile intervals",
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "accuracy": np.quantile(accuracy, [0.025, 0.975]).tolist(),
        "macro_f1": np.quantile(macro_f1, [0.025, 0.975]).tolist(),
        "scope": (
            "held-out sampling uncertainty, not training-seed or undetected-identity uncertainty"
        ),
    }
