"""Unit tests for src/ring_counting.py using synthetic images with a known answer.

The endodermis in the conftest prediction is a ring of radius ~0.10-0.14 * size.
We paint N evenly spaced bright Gaussian spots along it in the TRITC channel; the
peak counter should recover N.
"""

import numpy as np
import pytest

from src.ring_counting import (
    COUNT_PRESETS,
    RING_RANGE_PRESETS,
    multi_ring_peak_count,
    ring_peak_count,
    spacing_sanity_collapse_peaks,
    suggest_ring_range,
)

SIZE = 256
# Endodermis is r in [0.10, 0.14) * SIZE -> mid-ring ~0.12 * SIZE
RING_RADIUS = 0.12 * SIZE
PARAMS = dict(
    ring_width=4,
    n_points=720,
    smooth_sigma=2,
    peak_distance=20,
    peak_prominence=0.2,
    local_contrast_threshold=0,
    peak_width=1,
    merge_distance=10,
)


def _spotted_raw(n_spots, spot_sigma=2.0, size=SIZE):
    yy, xx = np.mgrid[:size, :size]
    raw = np.full((size, size, 3), 100.0, dtype=np.float32)
    for k in range(n_spots):
        a = 2 * np.pi * k / n_spots
        cy = size / 2 + RING_RADIUS * np.sin(a)
        cx = size / 2 + RING_RADIUS * np.cos(a)
        raw[..., 0] += 5000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * spot_sigma**2))
    return raw


@pytest.mark.parametrize("n_spots", [6, 10, 14])
def test_ring_peak_count_recovers_known_number_of_spots(bio7, n_spots):
    out = ring_peak_count(_spotted_raw(n_spots), bio7, "Endodermis", "TRITC", **PARAMS)
    assert out["count"] == n_spots
    assert len(out["peaks"]) == n_spots
    assert len(out["peak_points"]) == n_spots


def test_ring_peak_count_result_shape_and_invariants(bio7):
    out = ring_peak_count(_spotted_raw(8), bio7, "Endodermis", "TRITC", **PARAMS)
    assert len(out["signal"]) == PARAMS["n_points"]
    assert len(out["signal_smooth"]) == PARAMS["n_points"]
    assert out["channel"] == "TRITC"
    # Evenly spaced spots -> near-zero spacing variation, nothing to flag
    assert out["spacing_cv"] < 0.15
    assert out["mean_spacing"] == pytest.approx(PARAMS["n_points"] / 8, rel=0.05)
    assert isinstance(out["review_needed"], (bool, np.bool_))


def test_ring_peak_count_is_deterministic(bio7):
    raw = _spotted_raw(9)
    a = ring_peak_count(raw, bio7, "Endodermis", "TRITC", **PARAMS)
    b = ring_peak_count(raw, bio7, "Endodermis", "TRITC", **PARAMS)
    assert a["count"] == b["count"]
    np.testing.assert_array_equal(a["peaks"], b["peaks"])


def test_ring_peak_count_flat_signal_finds_nothing(bio7):
    raw = np.full((SIZE, SIZE, 3), 500.0, dtype=np.float32)
    out = ring_peak_count(raw, bio7, "Endodermis", "TRITC", **PARAMS)
    assert out["count"] == 0


def test_ring_peak_count_autofills_scale_dependent_params(bio7):
    # No ring_width / peak_distance / ... given: derived from root geometry.
    out = ring_peak_count(_spotted_raw(10), bio7, "Endodermis", "TRITC", n_points=720)
    assert out["count"] >= 1


def test_multi_ring_peak_count_agrees_with_truth(bio7):
    out = multi_ring_peak_count(
        _spotted_raw(10),
        bio7,
        "Endodermis",
        "TRITC",
        ring_position_min=0.3,
        ring_position_max=0.7,
        ring_position_iterations=3,
        ring_consensus_min=2,
        ring_width=4,
        n_points=720,
        peak_distance=20,
        peak_prominence=0.2,
        local_contrast_threshold=0,
        peak_width=1,
        merge_distance=10,
    )
    assert out["count"] == 10


class TestSpacingSanityCollapse:
    def test_even_spacing_is_untouched(self):
        peaks = np.arange(50, 1000, 100)
        _, info = spacing_sanity_collapse_peaks(np.ones(1000), peaks)[:2]
        assert info["applied"] is False
        assert info["removed"] == 0

    def test_duplicates_are_deduplicated(self):
        peaks = [100, 100, 300, 500, 700, 900]
        _, info = spacing_sanity_collapse_peaks(np.ones(1000), peaks)[:2]
        assert info["raw_count"] == 5


def test_presets_are_well_formed():
    assert set(COUNT_PRESETS) == {"Conservative", "Balanced", "Sensitive"}
    for preset in COUNT_PRESETS.values():
        assert {"min_spacing_ratio_trigger", "spacing_cv_trigger", "weak_peak_trigger"} <= set(
            preset
        )
    for lo, hi in RING_RANGE_PRESETS.values():
        assert 0 <= lo < hi <= 1


def test_suggest_ring_range_ranks_every_preset(bio7):
    # Regression: this used to raise NameError (enable_spacing_rescue / enable_valley_rescue
    # were referenced but never declared as parameters).
    df = suggest_ring_range(
        _spotted_raw(10),
        bio7,
        "Endodermis",
        "TRITC",
        ring_width=4,
        n_points=360,
        smooth_sigma=2,
        peak_distance=10,
        peak_prominence=0.2,
        local_contrast_threshold=0,
        peak_width=1,
        merge_distance=5,
        ring_consensus_min=2,
        ring_position_iterations=2,
    )
    assert len(df) == len(RING_RANGE_PRESETS)
    assert set(df["ring_range_label"]) == set(RING_RANGE_PRESETS)
    assert df["score"].is_monotonic_increasing  # best (lowest score) first
