# Changelog

## Unreleased — first measured experiment

- Audit the pinned Gano v3 collection, review similarity candidates, and freeze group-disjoint splits.
- Add optional augmentation, traced partial fine-tuning, validation-only checkpoint selection, and group-bootstrap intervals.
- Train and compare three fixed recipes; publish the selected model's 75.4% test accuracy, 0.751 macro F1 and 93.7% top-3 accuracy on 836 held-out images.
- Add standalone evaluation reports, scientific plots, opt-in local photo galleries, full experiment evidence, and runtime measurements.
- Document the source taxonomy, data limitations, actual results, reproducibility commands and model card.

### Review corrections

- Require an explicit training config path so installed CLI runs do not assume the checkout is the current directory.
- Record oversized-image decompression errors as data rejections without disabling Pillow's safety limit.
- Snapshot Git provenance before creating run outputs, with regression coverage for clean and dirty source checkouts.

## 0.1.0 — reproducible project foundation

- Package the original cat-breed experiment as a CLI with prepare, train, evaluate and predict commands.
- Add deterministic stratified manifests, image validation, exact deduplication and dataset drift checks.
- Embed MobileNetV2 normalization in saved models and persist the label order.
- Select checkpoints using validation loss; report held-out metrics and per-image scores explicitly.
- Save configuration, manifest, environment and Git provenance with each run.
- Add tests, CI, pinned dependencies and a complete lockfile.
- Preserve the original notebook as historical work and document data, architecture, experiment protocol and model limitations.

No new real-data accuracy result or pretrained cat model is included in this release.
