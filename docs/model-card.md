# Model card · Cat Vision's first measured model

## Status and intended use

A model was trained and evaluated on the checksum-pinned Gano v3 collection. The selected recipe is **frozen-horizontal-flip**. This is a hobby/portfolio classifier for the collection's visual labels. No public downloadable weight release or hosted prediction app is included yet; training commands reproduce the model locally.

Visual resemblance is not ancestry. Tuxedo is a coat pattern retained from the source taxonomy. Results describe these 15 source-provided labels, not biological breed verification.

## Model and training data

- ImageNet-pretrained MobileNetV2 with dropout 0.3 and a softmax head.
- Inputs: EXIF-aware RGB images, resized to 224×224; 0–255 pixels are normalized to -1–1 inside the saved model.
- Source: [Gano Cat Breed Image Collection](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection), version 3; attribution and stated collection terms are in the [source record](results/gano-v3/source.json).
- Audit: 5,625 decoded files, 52 redundant exact copies removed, 29 additional similarity pairs reviewed. Related-photo and filename-ID groups stay within one partition.
- Retained images: 3,896 training, 841 validation, 836 test across 15 labels.
- Within-run selection: lowest validation loss. Across-run selection: highest validation macro F1 from three planned recipes, before test predictions.

## Measured performance

| Metric | Selected model on the 836-image test split |
| --- | --- |
| Accuracy | 75.4% |
| 95% accuracy interval | 72.5%–78.0% |
| Macro F1 | 0.751 |
| 95% macro F1 interval | 0.722–0.777 |
| Top-3 accuracy | 93.7% |
| Training-majority baseline accuracy | 6.7% |

Intervals use 2,000 class-stratified group-bootstrap samples and describe held-out sampling uncertainty only. This is a single-seed study. Full per-label scores, confusion plots, validation comparison, checkpoint fingerprints, environment and timing are in the [experiment report](results/gano-v3/README.md).

The original 88.16% notebook figure was training accuracy with no independent test partition; it is not a comparison baseline for this score.

## Limits and unsuitable uses

Mixed-breed cats, unknown breeds, kittens, unusual poses, poor lighting and backgrounds can produce misleading predictions. There is no unknown/non-cat class or cat-localization stage. An unrelated image will still receive one of the known labels. Softmax scores are uncalibrated and do not establish certainty or breed purity.

The collection combines different sources and possibly noisy advertiser labels. Exact and reviewed visual deduplication do not establish complete animal-identity separation. Oxford/Google images lack reliable animal identifiers, and external/pretraining overlap has not been exhaustively audited. Source-specific backgrounds and watermarks may act as shortcuts. No independent phone-photo, unknown-breed or non-cat benchmark has been completed.

Do not use predictions for health, care, breeding, adoption eligibility or pedigree decisions. A high score within this curated collection is not a deployment-readiness claim.

## Artifacts and future release

The local run bundle includes the checkpoint, classes, configuration, grouped manifest, hashes, environment and metrics. Public evidence excludes raw dataset photographs and weight files. Confirm artifact distribution terms, include this card, and provide the full inference bundle before a future model release. Load only trusted artifacts.
