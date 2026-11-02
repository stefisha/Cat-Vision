# Verification record

This record describes software checks performed for the project refactor on **2026-11-02**. It does not report cat-breed recognition performance.

## Environment

- Apple Silicon macOS, CPU execution
- Python 3.11.18
- TensorFlow 2.21.0, Keras 3.15.1
- NumPy 1.26.4, Pillow 13.0.0, scikit-learn 1.6.1
- Ruff 0.11.2, pytest 8.3.5
- Full resolved dependencies: `uv.lock`

## Checks

| Check | Outcome |
| --- | --- |
| `ruff check .` | Pass |
| `ruff format --check .` | Pass |
| `pytest -q` | 19 tests pass, including the TensorFlow integration test |
| `cat-vision --help` | All four commands registered |
| CLI prepare → train → evaluate → predict | Pass on 18 synthetic images, three classes, two epochs, random frozen backbone |
| Build source archive and wheel with `uv build` | Pass |
| Python source compilation | Pass |
| Local Markdown link targets | Present |
| `git diff --check` | Pass |

The automated data tests check deterministic stratification, split disjointness, duplicate removal, conflicting labels, corrupt image reporting, changed file detection, manifest overwrite protection, invalid manifests, tiny datasets, RGB/pixel-scale conversion, and symlinks escaping the root. Invalid configurations fail before data access or run creation.

The TensorFlow integration test trains for two epochs on generated pixels without downloading ImageNet weights, reloads the checkpoint, evaluates the held-out fixture, verifies score/label contracts and normalization, checks repeated inference, and rejects altered labels, configuration, manifest, checkpoint fingerprints, and incomplete runs.

A separate subprocess smoke run exercises the installed CLI for each step. Its scores have no semantic meaning: the images are random pixels, not cats.

## Not established by these checks

- Generalization to cat photos, near-duplicate/source-group leakage, calibration, or dataset licensing/provenance.
- Real-data training time, prediction latency, GPU performance, Windows behavior, Intel macOS support, or Python 3.12 compatibility.
- A real-data baseline: the source images were not available in this workspace. Obtain the dataset, audit it, and follow the README's workflow before publishing any new performance claim.

GitHub Actions defines separate lightweight and TensorFlow CPU jobs on Ubuntu/Python 3.11, using the frozen dependency lock. Their current execution status is shown in the pull request and repository Actions tab; a local macOS pass does not substitute for a Linux CI pass.
