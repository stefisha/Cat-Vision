# Changelog

## 0.1.0 — reproducible project foundation

- Package the original cat-breed experiment as a CLI with prepare, train, evaluate and predict commands.
- Add deterministic stratified manifests, image validation, exact deduplication and dataset drift checks.
- Embed MobileNetV2 normalization in saved models and persist the label order.
- Select checkpoints using validation loss; report held-out metrics and per-image scores explicitly.
- Save configuration, manifest, environment and Git provenance with each run.
- Add tests, CI, pinned dependencies and a complete lockfile.
- Preserve the original notebook as historical work and document data, architecture, experiment protocol and model limitations.

No new real-data accuracy result or pretrained cat model is included in this release.
