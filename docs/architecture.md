# Architecture and decisions

## Data evidence and model evidence

```mermaid
flowchart LR
    A[Pinned source archive] --> B[Decode and content hashes]
    B --> C[Similarity candidates and filename IDs]
    C --> D[Reviewed photo groups]
    D --> E[Frozen group-disjoint manifest]
    E --> F[Training and validation]
    F --> G[Three planned recipes]
    G --> H[Validation-only selection record]
    H --> I[Selected checkpoint on held-out test]
    I --> J[Metrics, uncertainty, plots and local error gallery]
```

`audit.py` generates inspectable candidate evidence and connected image groups. `data.py` validates bytes and split membership. `model.py` defines the serialized network; `pipeline.py` trains, evaluates and predicts. `metrics.py` estimates held-out sampling uncertainty. `selection.py` freezes a choice from comparable validation results. `report.py` renders scientific plots and offline HTML. `cli.py` exposes these boundaries without importing TensorFlow for data preparation, selection or reporting.

## Key decisions

| Decision | Reason | Boundary |
| --- | --- | --- |
| Frozen ImageNet MobileNetV2 baseline | Small, understandable starting point consistent with the original project | Real performance must be measured |
| Partial fine-tuning as a separate child run | Preserve the parent baseline and trace inherited weights | Requires the same manifest, label map, image size and augmentation |
| BatchNorm frozen; backbone called with `training=False` | Preserve pretrained moving statistics during fine-tuning | Other eligible final backbone layers may train |
| Global average pooling, dropout, softmax head | A modest classifier with few head parameters | Different from the historical 256-unit dense head |
| Embedded `Rescaling(1/127.5, offset=-1)` | Keep the MobileNetV2 normalization contract with the model | Callers provide RGB pixels on the 0–255 scale |
| Shared Pillow orientation/RGB/resize path | Same pixel semantics in training and inference | EXIF and resizing live in Python, not the `.keras` graph |
| Shuffle file paths before decoding | Avoid buffering hundreds of megabytes of image tensors | Python-backed decoding favors clarity over maximum throughput |
| Reviewed similarity groups | Reduce observable duplicate-photo leakage | Hashes and filename IDs do not establish complete animal identity |
| Group-disjoint, class-preserving split | Related photos stay in one partition | Exact 70/15/15 proportions may be impossible for large groups |
| Lowest validation loss within each run | Predefined checkpoint selection | Validation can still be overused across many experiments |
| Highest validation macro F1 across three recipes | A documented label-balanced selection criterion | One seed and a small comparison, not a broad architecture search |
| Group-bootstrap intervals | Respect known dependence between related photos | Do not cover missed identities, training seeds or distribution shift |
| Standalone HTML, photos opt-in | Inspect metrics offline without a service | Embedding images creates a local file that needs source-rights review before sharing |
| JSON artifacts and local files | Inspectability without a platform dependency | No registry, experiment server or production deployment |

The preprocessing and fine-tuning choices follow [TensorFlow's MobileNetV2 API](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) and [transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).

## Integrity and reproducibility

The manifest records relative paths, labels, image hashes and groups. Training/evaluation rehash the image bytes. A run copies the manifest, settings and ordered classes; completion fingerprints the model and config. Loading verifies those artifacts and model shapes. The source Git revision and dirty flag are captured before any run output is created. Fine-tuning records its initial checkpoint hash.

Selection verifies completed checkpoints, configuration/manifest hashes, saved class mappings and a shared exact manifest. Evaluation metrics carry their own model/config/manifest fingerprints; selection rejects stale, unbound or copied scores. It reads validation metrics only and records their file hashes before writing an immutable selection record. These are guards against accidental artifact mixing, not authentication of untrusted files or a substitute for experimental discipline.

`uv.lock` fixes the resolved environment. Runs record the actual installed versions, Python, platform/devices, duration and source revision. Fixed seeds and deterministic TensorFlow operations reduce variability; floating-point results can still differ across hardware and software builds.

## Scope

There is no serving API or upload frontend yet. The current milestone is measured ML evidence and usable local inference. Infrastructure should follow a demonstrated deployment need.
