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

## Review corrections · 2026-11-03

Copilot identified three defects: a working-directory-dependent default training config, uncaught Pillow decompression-bomb errors, and a Git dirty flag captured after run files were created. Regression tests reproduced all three failures before the fixes.

Training now requires `--config` explicitly; tests invoke the CLI from outside the checkout and verify both the missing-argument diagnostic and loading an absolute config path. Preparation records `Image.DecompressionBombError` in `rejected` while leaving Pillow's safety limit enabled; a small real-image fixture exercises the error by temporarily lowering the threshold in the test only. Training captures source Git state before writing outputs; the TensorFlow integration test uses a temporary Git checkout and verifies both clean and pre-existing dirty states with outputs in a non-ignored directory.

After these changes, `pytest -q` reports **22 passed** on the same Python 3.11/Apple Silicon CPU environment. Ruff lint/format checks and `git diff --check` pass. These checks still make no claim about real-data model quality.

## Measured experiment · 2026-11-03

The real-data study follows `docs/experiment-plan.md`, committed before training. The archive checksum was verified, 29 additional similarity candidates were visually reviewed, and rerunning the CLI audit and group-aware preparation reproduced the exact committed manifest. It contains 3,896 training, 841 validation and 836 test images after removing 52 redundant exact files.

Three recipes ran on CPU from clean source states: frozen/no augmentation (16 epochs, selected epoch 12), frozen/horizontal flips (17 epochs, selected epoch 13), and partial fine-tuning (5 epochs, selected epoch 4). All use seed 42 and the same manifest. The core training/evaluation source stayed unchanged across their revisions. A separate commit records the validation-only selection before the selected checkpoint's test evaluation. The other two candidates have no test evaluation artifacts.

The expanded suite has 31 tests, covering audit/review freshness, group disjointness, deterministic bootstrap behavior, parent-checkpoint fine-tuning, frozen BatchNorm, validation/test output separation, selection integrity/tiebreaks, dataset extraction/checksums, report escaping, opt-in photo embedding and path safety, in addition to the original data and round-trip contracts. Linux quality and TensorFlow CI jobs pass. Scientific figures were rendered and visually inspected; aggregate documentation excludes dataset photographs.

Measured results are 75.4% test accuracy, 0.751 macro F1, and 93.7% top-3 accuracy. The 95% class-stratified group-bootstrap accuracy interval is 72.5–78.0%. The public results directory records configs, histories, environment, hashes, exact metrics and per-image predictions. A separate 50-image warm-model validation-image benchmark records CPU decode/resize plus inference timing; it is not a hosted-service latency test.

This completes a within-collection baseline, not independent phone-photo validation, a seed sweep, a complete animal-identity audit, calibration training or unknown/non-cat evaluation. The local weight bundle supports inference; weights and source photographs are not added to the public repository.

## Review integrity corrections · 2026-11-03

All six inline findings and the additional review-summary issues were checked. Visual decisions now require both originally reviewed image hashes. Evaluation metrics identify their model/config/manifest, and selection verifies those identities and the saved class map. Fine-tuning rejects inconsistent inherited initialization metadata. New downloads are checksum-verified before promotion, with failed partial transfers removed for a clean retry.

The stricter audit reproduced the original audit fingerprint and complete frozen manifest. The same three checkpoints were re-evaluated on validation and the selected checkpoint on test to add score-artifact identities: every numeric metric and prediction-file byte remained identical. The original pre-test selection record is untouched, and its original metric files are archived under `pre-review-evidence/`. `review-integrity-verification.json` records hashes and outcomes; no retraining or test-driven model change was performed.

The expanded suite passes **46 tests**, including stale positive/negative image judgments, copied and legacy metrics, saved class-map changes, retry after a bad HTTP-200 payload, oversized group allocation, initialization mismatch, optional-key diagnostics, accessible report tables and new CLI dispatch options. The committed CPU benchmark script was run successfully with explicit thread settings into a separate local output; the historical timing result was preserved. The confusion summary includes all pairs tied at seven, and the generic report includes the Tuxedo note only when that class is present.
