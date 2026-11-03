# Training and experiment protocol

The first measured study follows the [committed plan](experiment-plan.md), recorded before training. Its [results and evidence](results/gano-v3/README.md) separate validation-based selection from final test measurement.

## Reproduce the three recipes

Install `.[ml,report,dev]`, fetch the checksum-pinned dataset, and use the committed grouped manifest. Seed 42, RGB 224×224, dropout 0.3 and batch size 32 apply to all three runs.

```bash
cat-vision train \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --manifest docs/results/gano-v3/manifest.json \
  --config configs/frozen-no-augmentation.json \
  --run-dir runs/gano-v3/frozen-no-augmentation

cat-vision train \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --manifest docs/results/gano-v3/manifest.json \
  --config configs/frozen-horizontal-flip.json \
  --run-dir runs/gano-v3/frozen-horizontal-flip

cat-vision train \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --manifest docs/results/gano-v3/manifest.json \
  --config configs/partial-fine-tuning.json \
  --initial-run runs/gano-v3/frozen-no-augmentation \
  --run-dir runs/gano-v3/partial-fine-tuning
```

The two frozen runs use Adam 0.001, max 20 epochs and patience 4. The fine-tuning run starts from the no-augmentation checkpoint and trains eligible layers in the final 20 backbone layers at Adam 0.00001, max 5 epochs and patience 2. BatchNorm layers stay frozen and the backbone is called in inference mode. Image size, augmentation, labels and the exact manifest must match the parent run. The parent checkpoint hash is recorded.

`--config` is required; use absolute paths when running outside the checkout. `weights: null` supports no-download tests with a random frozen backbone; it is not a useful real-model recipe.

## Selection without test feedback

```bash
# Run separately for each candidate directory:
cat-vision evaluate \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --run-dir runs/gano-v3/frozen-no-augmentation \
  --split validation
```

After evaluating all three candidates on validation, freeze the decision:

```bash
cat-vision select \
  --run-dir runs/gano-v3/frozen-no-augmentation \
  --run-dir runs/gano-v3/frozen-horizontal-flip \
  --run-dir runs/gano-v3/partial-fine-tuning \
  --output runs/gano-v3/selection.json
```

Within a recipe, lowest validation loss selects the checkpoint. Compare the three saved checkpoints by validation macro F1; ties use lower validation log loss, then the simpler frozen model. Evaluation outputs bind scores to model, configuration and manifest SHA-256 values. Selection verifies all three fingerprints and the saved class mapping before ranking candidates. Legacy metric files without these identities must be regenerated with `evaluate --split validation`. Record the choice and checkpoint hashes before evaluating the selected run on test. The [selection record](results/gano-v3/selection.json) preserves this decision.

Evaluation defaults to `--split test`. The test partition is never used by `fit`, and unselected candidates were not evaluated on it in this study. This is one seed and three planned recipes, not a search for the highest test score.

## Visual reports

```bash
cat-vision report --run-dir runs/gano-v3/SELECTED_RECIPE

# Optional, local-only photo gallery:
cat-vision report \
  --run-dir runs/gano-v3/SELECTED_RECIPE \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --include-images \
  --output-dir runs/gano-v3/local-photo-report
```

Replace `SELECTED_RECIPE` with the selection record's name. The report includes learning curves, a row-normalized confusion heatmap with counts, per-label recall, a reliability diagram, a metrics table, and up to twelve of the most confidently incorrect predictions. The HTML embeds its figures and can be opened offline. Individual PNGs are also saved for use in documentation.

Dataset photos are omitted by default. `--include-images` embeds local thumbnails into the HTML; consider source-image rights before sharing that file. Neither mode uploads images anywhere. Error selection deliberately highlights confident mistakes and is not a representative sample of all images.

## Run bundle

| File | Purpose |
| --- | --- |
| `model.keras` | Best validation-loss checkpoint with pixel normalization |
| `classes.json`, `config.json` | Output label order and exact training settings |
| `manifest.json` | Image hashes, audited groups, split membership and exclusions |
| `provenance.json` | Data/config/checkpoint hashes, source Git state, environment, hardware and timing; parent hash for fine-tuning |
| `history.csv`, `history.json` | Per-epoch training/validation loss and accuracy |
| `validation_metrics.json`, `validation_predictions.json` | Candidate selection evidence |
| `metrics.json`, `test_predictions.json` | Selected model's held-out evidence |
| `test-report/index.html`, `*.png` | Regenerable standalone report and figures |

Keep bundles intact. Fingerprints detect accidental mixing of configurations, labels or checkpoints; they do not authenticate an untrusted artifact. Model files and raw images are ignored by Git. Training is deterministic where supported, but hardware/runtime differences can change results. Timing is machine-specific; recorded training duration excludes Python/TensorFlow import, final evaluation and report rendering.

## Metrics and uncertainty

Report accuracy, macro F1, top-3 accuracy, per-label precision/recall/F1/support, log loss, and the training-majority baseline. A confusion matrix uses actual labels in rows and predictions in columns. Macro F1 weights each label equally. Top-3 is meaningful here with 15 labels; with <=3 labels it is necessarily 100%.

The 95% intervals use 2,000 seeded class-stratified group-bootstrap samples. All images from a sampled audited group move together. Intervals describe sampling variation within the observed held-out groups; they do not include training-seed variation, missed identity overlap, label noise or distribution shift.

The reliability diagram evaluates uncalibrated softmax scores. No calibration model, unknown-breed detector or non-cat rejection mechanism has been trained.

## Historical evidence

The original notebook reached 88.16% **training** accuracy without a held-out split. It used a different classification head and data handling, so it is not directly comparable with this study's held-out score. Synthetic tests verify software contracts; they provide no evidence of breed-classification quality.


## Repeat the CPU inference benchmark

```bash
python scripts/benchmark_inference.py \
  --run-dir runs/gano-v3/frozen-horizontal-flip \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --samples 50 --warmups 5 --intra-threads 6 --inter-threads 1 \
  --output runs/gano-v3/runtime-repeat.json
```

The script sets TensorFlow thread counts explicitly before model initialization and uses evenly spaced validation images. It refuses to overwrite an existing measurement. Wall-clock results vary by hardware and current load; the historical `runtime.json` is retained as its original measurement. Historical training provenance does not record thread counts, so the explicit 6/1 configuration is attributed to the inference benchmark only.
