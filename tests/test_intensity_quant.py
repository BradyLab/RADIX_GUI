import numpy as np
import pytest

from src.intensity_quant import (
    measure_barrier_completeness,
    measure_root_morphometrics,
    normalize_channel,
    normalize_percentile_rgb,
)


def test_normalize_channel_range():
    img = np.linspace(0, 1000, 10_000, dtype=np.float32).reshape(100, 100)
    out = normalize_channel(img)
    assert out.min() == 0.0 and out.max() == 1.0
    assert out.dtype == np.float32


def test_normalize_channel_constant_image_is_zeros():
    assert not normalize_channel(np.full((10, 10), 5)).any()


def test_normalize_rgb_is_per_channel():
    ramp = np.arange(100.0).reshape(10, 10)
    out = normalize_percentile_rgb(np.stack([ramp, 1000 * ramp], axis=-1))
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-6)  # same shape, different scale


class TestMorphometrics:
    def test_areas_and_fractions(self, bio7):
        m = measure_root_morphometrics(bio7)
        root = float((bio7["Whole Root"] > 0).sum())
        assert m["root_area_px"] == pytest.approx(root)
        # Disk of radius 0.32 * 256 px
        assert m["root_diameter_px"] == pytest.approx(2 * 0.32 * 256, rel=0.03)
        assert 0.9 < m["root_circularity"] <= 1.1
        for t in ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular"]:
            assert m[f"{t}_area_px"] == (bio7[t] > 0).sum()
            assert m[f"{t}_frac"] == pytest.approx(m[f"{t}_area_px"] / root)

    def test_cortex_net_excludes_aerenchyma(self, bio7):
        m = measure_root_morphometrics(bio7)
        assert m["Cortex_net_area_px"] == m["Cortex_area_px"] - m["Aerenchyma_area_px"]
        assert 0 < m["Aerenchyma_cortex_frac"] < 1
        assert m["aerenchyma_pocket_count"] >= 1
        assert m["aerenchyma_inner_frac"] + m["aerenchyma_outer_frac"] == pytest.approx(1.0)
        assert m["cortex_stele_ratio"] > 0 and m["cortex_thickness_px"] > 0

    def test_micron_conversion(self, bio7):
        m = measure_root_morphometrics(bio7, um_per_px=0.5)
        assert m["root_area_um2"] == pytest.approx(m["root_area_px"] * 0.25)
        assert m["root_diameter_um"] == pytest.approx(m["root_diameter_px"] * 0.5)
        assert m["cortex_thickness_um"] == pytest.approx(m["cortex_thickness_px"] * 0.5)

    def test_no_micron_keys_without_calibration(self, bio7):
        assert not any(k.endswith(("_um", "_um2")) for k in measure_root_morphometrics(bio7))

    def test_empty_root_returns_nans(self):
        m = measure_root_morphometrics({"Whole Root": np.zeros((20, 20), np.uint8)})
        assert np.isnan(m["root_area_px"]) and np.isnan(m["cortex_stele_ratio"])

    def test_no_aerenchyma_gives_zero_pockets(self, bio7):
        no_aer = dict(bio7, Aerenchyma=np.zeros_like(bio7["Aerenchyma"]))
        m = measure_root_morphometrics(no_aer)
        assert m["aerenchyma_pocket_count"] == 0
        assert np.isnan(m["aerenchyma_mean_pocket_area_px"])


class TestBarrierCompleteness:
    """Completeness = fraction of angular sectors with above-threshold signal."""

    THR = {("Endodermis", "TRITC"): (500.0, np.inf)}

    @staticmethod
    def _raw_with_signal(bio7, angular_fraction):
        """TRITC bright only within `angular_fraction` of the circle."""
        h, w = bio7["Whole Root"].shape
        yy, xx = np.indices((h, w))
        ys, xs = np.where(bio7["Whole Root"] > 0)
        ang = np.arctan2(yy - ys.mean(), xx - xs.mean())
        lit = (ang > -np.pi) & (ang < -np.pi + 2 * np.pi * angular_fraction)
        raw = np.zeros((h, w, 3), dtype=np.float32)
        raw[..., 0] = np.where(lit, 1000.0, 0.0)
        return raw

    def test_full_ring_is_complete(self, bio7):
        out = measure_barrier_completeness(
            self._raw_with_signal(bio7, 1.0), bio7, self.THR, "Raw intensity", n_angles=60
        )
        assert out["Endodermis_TRITC_completeness"] == pytest.approx(1.0, abs=0.05)
        assert out["Endodermis_TRITC_completeness_pct"] == pytest.approx(
            100 * out["Endodermis_TRITC_completeness"]
        )

    def test_half_ring_is_about_half(self, bio7):
        out = measure_barrier_completeness(
            self._raw_with_signal(bio7, 0.5), bio7, self.THR, "Raw intensity", n_angles=60
        )
        assert out["Endodermis_TRITC_completeness"] == pytest.approx(0.5, abs=0.1)

    def test_no_signal_is_zero(self, bio7):
        raw = np.zeros((*bio7["Whole Root"].shape, 3), dtype=np.float32)
        out = measure_barrier_completeness(raw, bio7, self.THR, "Raw intensity", n_angles=60)
        assert out["Endodermis_TRITC_completeness"] == 0.0

    def test_empty_root_returns_empty_dict(self):
        empty = {"Whole Root": np.zeros((20, 20), np.uint8)}
        assert measure_barrier_completeness(np.zeros((20, 20, 3)), empty, {}, "Raw intensity") == {}
