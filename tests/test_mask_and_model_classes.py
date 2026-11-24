"""Unit tests for src/mask_utils.py and the semantic -> bio7 conversion."""

import numpy as np

from src.mask_utils import resize_bio7_to_raw
from src.model_classes import unet_semantic_to_bio7

BIO7_KEYS = {
    "Whole Root",
    "Epidermis",
    "Exodermis",
    "Cortex",
    "Aerenchyma",
    "Endodermis",
    "Vascular",
}


def test_resize_same_shape_is_passed_through():
    m = np.eye(8, dtype=np.uint8)
    out = resize_bio7_to_raw({"a": m}, (8, 8, 3))
    assert out["a"] is m


def test_resize_upscale_preserves_labels_and_shape():
    m = np.zeros((4, 4), dtype=np.uint8)
    m[:2, :2] = 1
    out = resize_bio7_to_raw({"a": m}, (8, 12, 3))  # raw is (H=8, W=12)
    assert out["a"].shape == (8, 12)
    # nearest-neighbour: no interpolated values, just the original label set
    assert set(np.unique(out["a"])) == {0, 1}
    assert out["a"][:4, :6].all() and not out["a"][4:, :].any()


def _sem():
    # one pixel per semantic label 0..6
    return np.arange(7, dtype=np.uint8).reshape(1, 7)


def test_bio7_keys_and_dtypes():
    bio7 = unet_semantic_to_bio7(_sem(), 1, 7)
    assert set(bio7) == BIO7_KEYS
    for m in bio7.values():
        assert m.dtype == np.uint8 and m.shape == (1, 7)
        assert set(np.unique(m)) <= {0, 1}


def test_bio7_label_mapping():
    bio7 = unet_semantic_to_bio7(_sem(), 1, 7)
    where = {k: {int(i) for i in np.flatnonzero(v[0])} for k, v in bio7.items()}
    assert where["Whole Root"] == {1, 2, 3, 4, 5, 6}  # everything but background
    assert where["Epidermis"] == {1}
    assert where["Aerenchyma"] == {2}
    assert where["Endodermis"] == {3}
    assert where["Vascular"] == {4}
    assert where["Exodermis"] == {5}


def test_bio7_cortex_includes_aerenchyma():
    bio7 = unet_semantic_to_bio7(_sem(), 1, 7)
    assert {int(i) for i in np.flatnonzero(bio7["Cortex"][0])} == {2, 6}
