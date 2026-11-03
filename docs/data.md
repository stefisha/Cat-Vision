# Data guide

## Versioned source

The first measured experiment uses [ShawnGano's Gano Cat Breed Image Collection](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection), Kaggle **version 3**, labeled “Version 1.1” by its author. It contains 5,625 images across 15 source-provided labels. Its source metadata lists Oxford pets, a PetFinder-derived collection, and a small Google Images contribution.

**Tuxedo is a coat pattern, not a breed.** We retain the supplied class for reproducibility and describe this experiment as classification of the collection's labels. Advertiser-provided PetFinder labels and combined sources introduce possible label noise and visual shortcuts.

The collection page states **CC BY-SA 4.0**. This is recorded attribution, not an independent verification of every underlying photograph. The repository does not redistribute dataset images. Code/artwork licensing is separate. See the [source record](results/gano-v3/source.json).

```bash
python scripts/fetch_gano.py
```

The downloader pins the version, verifies archive SHA-256 `4af0320b13d0dff4cf2ab332557d7d0eba60624115779e854968dad25ba40c15`, validates extraction paths, and refuses to overwrite an extracted dataset. The 266,219,297-byte archive and images live under ignored `data/` directories. No Kaggle credentials are needed for this public version at the time of the experiment; access can change.

## Input contract

- One root, one immediate subdirectory per label, supported image files inside each subdirectory.
- JPEG, PNG, BMP, GIF and WebP suffixes. Nested folders and non-image suffixes are ignored.
- Pillow fully decodes pixels, corrects EXIF orientation, converts to RGB and applies bilinear resizing. Animated images use the first frame; transparent images are converted without background compositing.
- Unreadable images and Pillow decompression-bomb errors are recorded as rejections; the pixel safety limit remains enabled.
- At least two labels and three unique valid images per label. Audited preparation also requires at least three independent groups per label.

## Audit before splitting

```bash
cat-vision audit \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --output data/audits/gano-v3.json \
  --decisions docs/results/gano-v3/review-decisions.json

cat-vision prepare \
  --data-dir data/raw/Gano-Cat-Breeds-V1_1 \
  --audit data/audits/gano-v3.json \
  --manifest data/manifests/gano-v3.json
```

For the pinned collection, the reviewed decisions are committed. For a new dataset, first run `audit` without `--decisions`. Review the `similar_pairs` and provide a JSON list with `left`, `right`, `left_sha256`, `right_sha256`, boolean `same_photo`, and a short `note` for each candidate. Copy both hashes from the original audit's `records` when reviewing the pair. Missing or changed hashes invalidate a decision, including a previous `same_photo: false`; inspect changed images again rather than blindly refreshing their hashes. Rerun audit into a new output path. Unreviewed, duplicate, or stale decisions cannot be used silently for grouped preparation.

Candidate policy: 64-bit DCT perceptual hash distance <=6 **or** 64-bit difference hash distance <=4. Those are screening thresholds, not proof of a duplicate. Exact decoded RGB pixels, filename PetFinder listing IDs, and approved similarity pairs form connected groups. Filename-inferred source families are diagnostics; unmatched names are `unknown`. They are not verified capture provenance or a complete animal-identity system.

Preparation removes byte-identical redundant files, verifies that the audit matches current bytes, and keeps each connected group in exactly one split. Groups spanning conflicting labels stop preparation for review. Within each label, seeded largest-first allocation targets approximately 70/15/15 image proportions while preserving all three partitions. A repair pass considers whole-group moves/swaps when the initial holdouts are oversized, while preserving class coverage. Group sizes can still prevent exact percentages. Already balanced partitions, including the frozen study manifest, remain unchanged. Without `--audit`, the original image-level split remains available for exploration; it does not provide the stronger grouping guarantee used by the measured experiment.

## Audit outcome

- **5,625** valid decoded images; **0** decode rejections with the current Pillow pipeline.
- **50** exact-duplicate clusters; **52** redundant exact files removed.
- **29** additional similarity candidates visually reviewed: **28** approved photo-copy pairs and **1** false match.
- Sources inferred from naming: 3,204 PetFinder, 2,374 Oxford, 47 Google images before deduplication.
- **5,573** retained images: **3,896 train / 841 validation / 836 test**.

The [manifest](results/gano-v3/manifest.json), [review decisions](results/gano-v3/review-decisions.json) and [audit summary](results/gano-v3/audit-summary.json) make this membership inspectable. The original notebook's 5,624 count reflected its manual exclusion; the current decoder accepts all 5,625 source images. Do not silently substitute notebook counts for this audit.

## Remaining leakage limits

Perceptual hashes miss some crops, viewpoint changes and different photographs of the same animal. Oxford/Google filenames do not provide reliable animal IDs. The audit reduces observed leakage but does not establish complete identity independence. No independent collection of phone photos has been evaluated, and pretrained-dataset overlap has not been exhaustively checked.

Use validation data to choose experiments. If test errors inform later changes, treat that test split as development evidence and obtain a new independent final holdout. A CLI flag cannot enforce experimental discipline.

Training and evaluation rehash manifest images and reject changed/missing bytes or paths outside the root. Extra files are excluded until a new manifest is prepared. Audits, manifests and run destinations are never silently overwritten.
