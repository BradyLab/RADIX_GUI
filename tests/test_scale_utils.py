"""Unit tests for src/scale_utils.py (pure numpy/skimage, no GUI)."""

import numpy as np
import pytest

from src.scale_utils import (
    MIN_CELL_SCALE_PX,
    MIN_ROOT_DIAMETER_PX,
    compute_root_scale,
    describe_scale,
    scaled_blob_defaults,
    scaled_ring_defaults,
)


def _disk(size, radius):
    yy, xx = np.mgrid[:size, :size]
    return (np.hypot(yy - size / 2, xx - size / 2) < radius).astype(np.uint8)


def test_root_diameter_matches_disk():
    scale = compute_root_scale({"Whole Root": _disk(1000, 400)})
    assert scale["valid"] is True
    assert scale["source"] == "root_geometry"
    assert scale["root_diameter_px"] == pytest.approx(800, rel=0.02)
    # 1.2% of root diameter
    assert scale["cell_scale_px"] == pytest.approx(0.012 * scale["root_diameter_px"])


def test_largest_component_wins_over_stray_pixels():
    mask = _disk(1000, 400)
    mask[5:8, 5:8] = 1  # stray prediction noise, far from the root
    scale = compute_root_scale({"Whole Root": mask})
    assert scale["root_diameter_px"] == pytest.approx(800, rel=0.02)


def test_physical_scale_uses_microns_per_pixel():
    scale = compute_root_scale(
        {"Whole Root": _disk(1000, 400)}, um_per_pixel=0.5, typical_cell_um=12
    )
    assert scale["source"] == "physical"
    assert scale["cell_scale_px"] == pytest.approx(24.0)


def test_cell_scale_floor_applies():
    # Huge pixels -> tiny cell, but never below the floor.
    scale = compute_root_scale({"Whole Root": _disk(1000, 400)}, um_per_pixel=100.0)
    assert scale["cell_scale_px"] == MIN_CELL_SCALE_PX


def test_small_root_diameter_is_floored():
    scale = compute_root_scale({"Whole Root": _disk(100, 10)})
    assert scale["root_diameter_px"] == MIN_ROOT_DIAMETER_PX


@pytest.mark.parametrize("bio7", [{}, {"Whole Root": np.zeros((50, 50), np.uint8)}])
def test_missing_or_empty_mask_falls_back(bio7):
    scale = compute_root_scale(bio7)
    assert scale["valid"] is False
    assert scale["source"] == "fallback_empty_mask"
    assert scale["cell_scale_px"] == MIN_CELL_SCALE_PX


def test_scaled_defaults_grow_with_cell_scale():
    small = {"cell_scale_px": 10.0}
    large = {"cell_scale_px": 40.0}
    for fn in (scaled_ring_defaults, scaled_blob_defaults):
        s, lg = fn(small), fn(large)
        assert set(s) == set(lg)
        assert all(lg[k] >= s[k] for k in s)
        assert all(isinstance(v, int) for v in s.values())


def test_scaled_defaults_have_floors():
    tiny = {"cell_scale_px": 0.1}
    ring = scaled_ring_defaults(tiny)
    assert ring["peak_distance"] >= 3 and ring["ring_width"] >= 2
    assert ring["merge_distance"] >= 4 and ring["peak_width"] >= 1
    blob = scaled_blob_defaults(tiny)
    assert blob["min_area"] >= 20 and blob["max_area"] >= 2000 and blob["min_distance"] >= 3


def test_describe_scale_is_one_line():
    text = describe_scale(compute_root_scale({"Whole Root": _disk(1000, 400)}))
    assert "\n" not in text
    assert "root_geometry" in text and "Cell scale" in text
