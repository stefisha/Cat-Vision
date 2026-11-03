# First measured experiment: fixed plan

Recorded before training or evaluating models on the downloaded Gano v3 collection.

## Dataset

Use the original public Gano collection, Kaggle version 3. Archive SHA-256: `4af0320b13d0dff4cf2ab332557d7d0eba60624115779e854968dad25ba40c15`. Retain all 15 dataset labels, including Tuxedo (a coat pattern, not a breed). Remove byte-identical redundant files. Group decoded-pixel matches, repeated filename PetFinder identifiers, and visually approved similarity candidates before splitting.

Similarity candidate policy is pHash Hamming distance <=6 OR dHash distance <=4. Review all candidates before freezing split membership. Do not tune the grouping policy using model performance. Visually rejected false matches remain separate. An image/identity group cannot cross splits. Class coverage and approximate 70/15/15 image proportions are required; actual grouped sizes will be reported.

## Three planned recipes

1. Frozen ImageNet MobileNetV2, no augmentation: 224px, Adam 0.001, dropout 0.3, batch 32, max 20 epochs, patience 4.
2. Frozen ImageNet MobileNetV2, horizontal flips: same settings and seed; only augmentation differs.
3. Partial fine-tuning of recipe 1's best validation-loss checkpoint: final 20 backbone layers eligible to train, BatchNorm frozen, Adam 0.00001, max 5 epochs, patience 2; no augmentation.

Seed 42 and the same audited manifest apply to all recipes. This is a small single-seed comparison, not an exhaustive hyperparameter search. No winner or improvement is assumed.

## Selection and final measurement

Within a recipe, choose the checkpoint with lowest validation loss. Across the three saved checkpoints, choose the largest validation macro F1 (ties: lower validation loss, then simpler frozen model). Write a selection record with candidate checkpoint hashes and validation scores **before** computing test predictions. Do not evaluate other candidates on the test set, and do not change recipes in response to the selected model's test errors.

Report held-out accuracy, macro F1, top-3 accuracy, per-label precision/recall/F1/support, confusion matrix, and the training-majority baseline. Add 95% percentile intervals from 2,000 seeded class-stratified group-bootstrap samples. These describe uncertainty in the observed held-out sample, not variation across training seeds or all possible cats.

Generate learning curves, validation comparisons, per-label recall and confusion heatmaps. An error gallery is a post-selection diagnostic; it does not become a new source of tuning evidence in this experiment. Dataset photos stay local; published reports use aggregate figures and image paths rather than redistributing photographs.

## Limits

Perceptual hashes do not establish complete animal identity. Filename identifiers are only available for one source. Labels are source-provided, with known taxonomy and potential annotation noise. Source and pretrained-dataset overlap have not been exhaustively audited. The test split estimates performance within this collection; it is not a benchmark on independent phone photos, unknown breeds or non-cat images.
