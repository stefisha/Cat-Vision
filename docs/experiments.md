# Training and experiment protocol

## Baseline configuration

`--config` is required so installed CLI runs never depend on an implicit path in the current working directory. Use an absolute path when invoking training from another directory.

`configs/baseline.json` is the initial recipe: seed 42, 224×224 RGB images, batches of 32, Adam at 0.001, a maximum of 20 epochs, and patience 4. Training uses sparse categorical cross-entropy. A frozen ImageNet MobileNetV2 backbone feeds dropout 0.3 and a softmax classifier. Horizontal flips apply during training only.

Validation loss selects the best checkpoint and drives early stopping. The test split is only used by the explicit evaluation command. There is no automatic fine-tuning stage; start with an understandable frozen-feature baseline.

`weights: null` avoids a pretrained download for tests. The backbone remains frozen in this mode: it is a plumbing check, **not a model-training recipe**. Use `"imagenet"` for the real baseline.

## Run bundle

| File | Purpose |
| --- | --- |
| `model.keras` | Best validation-loss checkpoint, including pixel normalization |
| `classes.json` | Ordered labels; output column `i` corresponds to label `i` |
| `config.json` | Training settings |
| `manifest.json` | Dataset membership, hashes, split policy, exclusions |
| `provenance.json` | Manifest/config/checkpoint fingerprints, completion status, Python/packages, Git revision and dirty flag, platform/devices, duration, best epoch |
| `history.csv`, `history.json` | Training and validation loss/accuracy over epochs |
| `metrics.json` | Held-out metrics and confusion matrix after evaluation |
| `test_predictions.json` | Test image path, actual label, prediction and class scores |

Keep the entire bundle together. The classifier alone cannot communicate label names or EXIF/resize semantics. Artifact directories and raw data are ignored by Git. For a future release, attach the model bundle only after reviewing distribution terms, and publish the data manifest and measured results with it.

## Measurement protocol

1. Audit the data and freeze a manifest. Commit the code used for the run.
2. Train the baseline. Inspect loss curves for underfitting or overfitting using training and validation data.
3. Record one proposed change and why it might help. Compare runs on the **same validation split**.
4. Select the final configuration using validation evidence. Evaluate the test partition after selection.
5. Report accuracy, macro F1, per-breed precision/recall, confusion matrix, and training-majority baseline. Include dataset size, exclusions, class order, split fingerprint, revision, environment and hardware.
6. Inspect representative errors, near-duplicates and out-of-distribution photos. Describe limitations alongside successes.

Top-3 accuracy includes up to three labels. For a two- or three-class dataset, it is necessarily 100% and is not informative. Softmax scores are rankings, not calibrated confidence.

## Results ledger

| Experiment | Evidence | Result | Interpretation |
| --- | --- | --- | --- |
| Original Colab notebook | Saved notebook training log | 88.16% training accuracy | No validation/test split; not a generalization estimate |
| Refactored pipeline smoke test | Synthetic generated images, two epochs, no pretrained weights | See CI / verification notes | Tests execution and artifact contracts only |
| Real-data frozen-backbone baseline | Pending dataset audit and execution | **Not measured** | No portfolio performance claim yet |

When adding a real run, replace pending entries with measured values and a link to a reproducible run record. Do not promote synthetic smoke-test accuracy as cat recognition performance.

## A useful experiment note

Copy these fields into a new note under `docs/results/` when a real run is complete:

- Question and hypothesis
- Git revision; clean/dirty state; hardware; wall-clock duration
- Manifest SHA-256; source dataset version; counts per class and split; exclusions
- Configuration and environment lock revision
- Validation selection rationale and previous experiments considered
- Held-out metrics and baseline; confusion matrix and representative errors
- Leakage audit, limitations, and next experiment
