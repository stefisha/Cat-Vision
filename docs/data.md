# Data guide

## Source and responsibility

The original experiment uses the [Gano Cat Breed Image Collection](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection). Its saved notebook output reports 5,624 images across 15 breed folders. Those counts describe the historical local copy; they are not a guarantee about a current download.

Download manually and review the dataset page, license, attribution, and image provenance. The repository's MIT license covers code, not dataset photos. No images, Kaggle credentials, or trained weights are committed. Do not add an automated download until access and licensing requirements are understood.

## Input contract

- One root directory, one immediate subdirectory per breed, image files inside each breed folder.
- Supported suffixes: JPEG, PNG, BMP, GIF, and WebP. Labels are folder names.
- At least two classes, each with at least three unique decodable images.
- Files are decoded with Pillow, corrected for EXIF orientation, converted to RGB, and resized with bilinear interpolation. Animated images use the first frame. Transparent images are converted to RGB without background compositing.
- Class discovery and file traversal are sorted. Non-image suffixes and nested subfolders are ignored.

## Preparation policy

1. Decode supported images. Record unreadable images and their error messages, including images rejected by Pillow's decompression-bomb error. Pillow's pixel safety limit remains enabled.
2. Hash file bytes with SHA-256. Keep one instance of an identical image within a breed. Stop on identical bytes under conflicting labels.
3. Shuffle each sorted breed group with a seeded local random generator.
4. Allocate `max(1, round(0.15 * class_count))` each to test and validation; use the rest for training. Python's rounding applies. Tiny classes will not have a 70/15/15 ratio.
5. Save the manifest. Refuse to overwrite an existing one.

Review `rejected` and `duplicates` before continuing. Preparation skips corrupt images to make the issue inspectable; it does not silently repair them. A class with no valid images is absent from the discovered labels, so check the final class list against the source folders.

## Leakage and dataset revisions

Byte hashes detect exact copies only. Re-encoded copies, crops, near-duplicates, photos of the same cat, watermarks and source-specific backgrounds can cross partitions. Before publishing results, inspect perceptual similarity and source/animal identity where available; if groups can be established, implement a group-aware split and record that policy as a new manifest version.

Never tune using test mistakes. If test feedback influences the next model, treat that split as development data and obtain a fresh final holdout. Repeatedly executing evaluation cannot enforce methodological independence; the experimenter must preserve it.

Training and evaluation rehash every manifest image and fail on changed bytes, missing files, or paths outside the root. Extra files added later are not included in a frozen manifest. Use a new manifest to include them. Run directories copy the manifest so moving the dataset root is safe as long as relative paths and bytes are preserved.
