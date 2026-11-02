# Model card · Cat Vision baseline recipe

## Status

This is a model **recipe and evaluation pipeline**. No trained model or verified held-out breed-classification result is released. The historical notebook's 88.16% training accuracy is not evidence of performance on new photos.

## Intended use

Personal learning, portfolio discussion, and exploratory breed-image classification. Input is a photo; output is a ranked list from the breed labels in the training manifest. The original dataset contained 15 labels, but the pipeline supports a discovered label set.

## Model and data

- ImageNet-pretrained MobileNetV2, frozen feature extractor with a small dropout/softmax head.
- Inputs: EXIF-aware RGB images resized to 224×224, raw 0–255 values. Normalization to -1–1 is embedded in the model.
- Training source: user-supplied Gano collection, subject to its own terms and provenance.
- Data split: seeded, stratified, approximately 70/15/15 per class after exact deduplication.
- Model selection: lowest validation loss. Final evaluation: saved checkpoint on the held-out test subset.

## Evaluation to fill after a real run

| Field | Status |
| --- | --- |
| Dataset version and audited counts | Pending |
| Manifest fingerprint and code revision | Pending |
| Test accuracy / macro F1 / top-3 | Not measured |
| Per-class metrics and confusion matrix | Not measured |
| Majority-class comparison | Not measured |
| Calibration and out-of-distribution evaluation | Not implemented |
| Hardware, inference latency and training cost | Not benchmarked |

## Limitations

Visual resemblance is not ancestry. Mixed-breed cats, unknown breeds, kittens, unusual poses, poor lighting and backgrounds may produce misleading predictions. The model has no explicit unknown or non-cat class and does not localize a cat in the image. It returns a known class even when the image is unrelated.

Softmax scores are not calibrated estimates. Do not interpret a score as certainty, breed purity, or a biological fact. Do not use predictions for health, care, breeding, adoption eligibility, or commercial pedigree claims.

Exact deduplication does not remove near-duplicates or same-animal overlap. Source and breed labels may be noisy; backgrounds and watermarks can be shortcuts. Breed balance alone does not establish representativeness across age, geography, image source or capture conditions. Document those gaps when reporting performance.

## Release checklist

Before distributing a model: complete the data and leakage audit, measure a fixed holdout, fill this card with evidence, verify distribution rights for weights and data artifacts, and document the full inference bundle. Only load model artifacts from trusted sources.
