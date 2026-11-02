import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "cats"
    rng = np.random.default_rng(7)
    for label in ("Siamese", "Bengal", "Maine Coon"):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(12):
            Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(
                folder / f"cat-{i:02}.png"
            )
    return root
