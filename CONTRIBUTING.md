# Contributing

Cat Vision is a small hobby project with an emphasis on readable experiments. Focused fixes, clearer explanations, and reproducible ML improvements are welcome.

## Local setup

Use Python 3.11. Install `.[ml,dev]` in a virtual environment or use `uv sync --frozen --extra ml --extra dev`. Run `ruff check .`, `ruff format --check .`, and `pytest -q` before opening a pull request. CI runs the same checks using generated image fixtures; Kaggle access is never required.

## Changes to ML behavior

Explain the hypothesis and which data, preprocessing, label mapping, or metric semantics change. Add regression tests where contracts could break. Report actual commands and results. A unit test passing is evidence of software behavior, not model accuracy.

Use validation data for experiments, keep the final test set independent, and create a new run directory for each experiment. Commit new configs rather than editing old run artifacts. Update the data guide/model card if the assumptions change.

## Dependencies

Direct versions are pinned in `pyproject.toml`; regenerate `uv.lock` with `uv lock` after an intentional update and run both the lightweight and TensorFlow tests. Dependency updates should be separate from model-quality comparisons when possible. The historical notebook is excluded from formatting and linting so its saved experiment remains intact.

## Issues and pull requests

Include exact reproduction commands, Python/OS versions, relevant logs, and whether data is real or synthetic. Avoid credentials, personal images, datasets or generated model files in commits. Use a small synthetic fixture to reproduce a bug where possible.
