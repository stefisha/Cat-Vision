<p align="center">
  <img src="assets/banner.svg" alt="Cat Vision — curiosity meets computer vision" width="100%">
</p>

<p align="center">
  <a href="https://github.com/stefisha/Cat-Vision/actions/workflows/ci.yml"><img src="https://github.com/stefisha/Cat-Vision/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB" alt="Python 3.11 and 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00" alt="TensorFlow 2.21.0">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-79d9d0" alt="Code license: MIT"></a>
</p>

I like cats. I also like understanding what a model has actually learned. **Cat Vision** brings those interests together: a small, reproducible project that classifies cat photos by breed using MobileNetV2 transfer learning.

It started as a Colab notebook. The current project adds the engineering needed to make an experiment inspectable: validated data, frozen splits, consistent preprocessing, traceable runs, held-out evaluation, and automated tests. The goal is a thoughtful portfolio project with a modest footprint.

> **Results status:** the original notebook recorded **88.16% training accuracy** across 15 breeds and 5,624 images. It had no held-out evaluation. The refactored pipeline has been tested on synthetic images; a real-data baseline has **not yet been measured**. No trained breed classifier or dataset is distributed here.

## What is here

- **Data integrity:** decode checks, byte-level deduplication, conflicting-label detection, SHA-256 image fingerprints, and a persisted stratified split.
- **Training:** a frozen ImageNet MobileNetV2 backbone, train-only horizontal flips, embedded pixel normalization, validation-loss early stopping, and best-checkpoint saving.
- **Evaluation:** held-out accuracy, macro F1, top-3 accuracy, per-breed metrics, confusion matrix, per-image scores, and a training-majority baseline.
- **Reproducibility:** one class mapping, seeded runs, configuration snapshots, dependency lockfile, package versions, and Git provenance.
- **Quality:** a packaged CLI, data contract tests, a no-download TensorFlow integration test, and GitHub Actions.

## Quick start

Use **Python 3.11** for the tested environment. Python 3.12 is allowed by the package; it is not in the CI matrix yet. The target platforms are Linux x86-64 and Apple Silicon macOS 12+. Intel macOS is not supported by this TensorFlow pin; use Linux or Colab there. CPU execution works; a GPU is optional. For accelerator setup, follow the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

```bash
git clone https://github.com/stefisha/Cat-Vision.git
cd Cat-Vision
python3.11 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[ml,dev]'
cat-vision --help
```

For a fully resolved environment, use [uv](https://docs.astral.sh/uv/): `uv sync --frozen --extra ml --extra dev`, then run commands with `uv run --frozen --extra ml --extra dev`. The pip installation pins direct dependencies; `uv.lock` also fixes transitive dependencies. CI installs from the lockfile.

### 1. Bring your own cats

Download and extract the [Gano Cat Breed Image Collection](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection). Review its current terms and image provenance before use or redistribution. Kaggle credentials and data stay out of Git.

Point `--data-dir` at the directory whose **immediate subdirectories are breed labels**:

```text
data/raw/Gano-Cat-Breeds-V1_1/
├── Abyssinian/
│   ├── cat-001.jpg
│   └── ...
├── Bengal/
└── ...
```

The pipeline discovers the labels; it does not hard-code 15 breeds. Each class needs at least three unique valid images. That minimum supports smoke testing, not a credible benchmark.

### 2. Validate and freeze the split

```bash
cat-vision prepare \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --manifest data/manifests/v1.json
```

This creates approximately **70% train / 15% validation / 15% test per class** using seed 42. Review the manifest's `rejected` and `duplicates` lists before training. Existing manifests are never silently overwritten. Training and evaluation verify file hashes, so modifying an image requires a new manifest and a new experiment.

Exact duplicates are removed before splitting. Near-duplicates and multiple photos of the same cat can still leak between splits; audit these before publishing performance claims. See the [data guide](docs/data.md).

### 3. Train a baseline

```bash
cat-vision train \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --manifest data/manifests/v1.json \
  --config configs/baseline.json \
  --run-dir runs/baseline-v1
```

`--config` is required; pass an absolute config path when running outside the checkout. The first baseline run downloads ImageNet weights. Validation loss selects `model.keras`; the test set is never supplied to `fit`. Run directories are immutable destinations: choose a new name for each experiment. See [training and experiments](docs/experiments.md) for the rationale and run artifacts.

### 4. Evaluate once the experiment is selected

```bash
cat-vision evaluate \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --run-dir runs/baseline-v1
```

Read `metrics.json` and `test_predictions.json` in the run directory. The confusion matrix uses the `classes` order, with **actual labels in rows and predictions in columns**. Compare macro F1 and per-breed recall with overall accuracy; inspect mistakes rather than selecting only attractive examples.

### 5. Try an image

```bash
cat-vision predict \
  --run-dir runs/baseline-v1 \
  --image /path/to/your/cat.jpg \
  --top-k 3
```

The output is ranked breed scores. A softmax score is not a calibrated probability of ancestry. The model always chooses among its known labels—even for a mixed-breed cat, a dog, or an empty photo. See the [model card](docs/model-card.md).

## Project map

```text
src/cat_vision/       Data contracts, model, pipeline, CLI
configs/             Versioned experiment settings
notebooks/           Original exploratory experiment (historical)
tests/               Data tests and TensorFlow round-trip smoke test
docs/                Data guide, decisions, experiments, model card
assets/              Repository artwork
.github/             CI and contribution templates
uv.lock              Resolved dependency versions
```

## Development

```bash
# With the activated environment from above:
ruff check .
ruff format --check .
pytest -q
```

The test suite creates tiny synthetic images and uses `weights=None`; it needs neither Kaggle data nor a pretrained-weight download. It verifies plumbing, not model quality. The lightweight subset can run without TensorFlow: install `'.[dev]'`, then `pytest -m 'not ml' -q`.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for changes, [architecture](docs/architecture.md) for design decisions, and [verification](docs/verification.md) for the checks performed during this refactor.

## Next experiments

1. Audit near-duplicates and source groups, then publish a versioned real-data baseline.
2. Compare gentle fine-tuning against the frozen backbone using validation data only.
3. Examine calibration, class-specific errors, and photos outside the training distribution.
4. Add a small inference demo once a documented model is available.

## Credits and license

Dataset: [Gano Cat Breed Image Collection](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection). Backbone: [MobileNetV2](https://arxiv.org/abs/1801.04381), via [Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2). Original experiment: [historical notebook](notebooks/original_experiment.ipynb).

Code and repository artwork are under the [MIT license](LICENSE). Dataset images and third-party model weights retain their own terms. This is a hobby and learning project, not a tool for establishing a cat's ancestry or making care decisions.
