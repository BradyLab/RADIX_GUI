import numpy as np
import pytest

from src.preprocessing import (
    normalize_percentile,
    resize_with_padding,
    to_uint8,
    unpad_predictions,
)


def test_normalize_percentile_range_and_independent_channels():
    rng = np.random.default_rng(0)
    img = np.stack(
        [rng.uniform(0, 100, (32, 32)), rng.uniform(5000, 9000, (32, 32))], axis=-1
    ).astype(np.float32)
    out = normalize_percentile(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Channels are normalised on their own scale, so both should span ~[0, 1]
    for c in range(2):
        assert out[..., c].min() == 0.0 and out[..., c].max() == 1.0


def test_normalize_constant_channel_is_zero_not_nan():
    out = normalize_percentile(np.full((8, 8, 1), 42.0, dtype=np.float32))
    assert not np.isnan(out).any()
    assert (out == 0).all()


def test_to_uint8_clips():
    out = to_uint8(np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32))
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 127, 255, 255]


def test_resize_with_padding_geometry():
    img = np.ones((100, 200, 3), dtype=np.float32)
    padded, masks, meta = resize_with_padding(img, target_size=64)
    assert masks is None
    assert padded.shape == (64, 64, 3)
    assert (meta.scaled_h, meta.scaled_w) == (32, 64)
    assert (meta.pad_bottom, meta.pad_right) == (32, 0)
    # padding is zeros at the bottom, content above
    assert padded[:32].min() == 1.0 and padded[32:].max() == 0.0


def test_mask_round_trip_through_pad_and_unpad():
    img = np.zeros((90, 120, 3), dtype=np.float32)
    masks = np.zeros((2, 90, 120), dtype=np.uint8)
    masks[0, 20:60, 30:90] = 1
    masks[1, 0:10, 0:10] = 1
    _, padded_masks, meta = resize_with_padding(img, target_size=60, masks=masks)
    assert padded_masks.shape == (2, 60, 60)

    restored = unpad_predictions(padded_masks, meta)
    assert restored.shape == (2, 90, 120)
    for i in range(2):
        # nearest-neighbour round trip should agree with the original almost everywhere
        assert (restored[i] == masks[i]).mean() > 0.97

    single = unpad_predictions(padded_masks[0], meta)
    assert single.shape == (90, 120)
    np.testing.assert_array_equal(single, restored[0])


@pytest.mark.parametrize("target", [32, 128])
def test_scale_is_target_over_longest_side(target):
    _, _, meta = resize_with_padding(np.zeros((50, 80, 3), np.float32), target_size=target)
    assert meta.scale == pytest.approx(target / 80)
