import numpy as np
import pytest
import tifffile

from src.io_utils import infer_species, load_raw_tiff_sample, load_sample


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Sorghum_exp3_Sample_01", "Sorghum"),
        ("RICE_plate2", "Rice"),
        ("millet-12", "Millet"),
        ("tomato_A", "Tomato"),
        ("arabidopsis_1", "Unknown"),
    ],
)
def test_infer_species(name, expected):
    assert infer_species(name) == expected


def test_load_raw_stacks_channels_in_requested_order(sample_data_dir):
    raw = load_raw_tiff_sample("Sample_01", sample_data_dir, ["TRITC", "FITC", "DAPI"])
    assert raw.shape == (256, 256, 3)
    assert raw.dtype == np.uint16
    # Each channel is a different random image, so order matters.
    swapped = load_raw_tiff_sample("Sample_01", sample_data_dir, ["DAPI", "FITC", "TRITC"])
    np.testing.assert_array_equal(raw[..., 0], swapped[..., 2])
    np.testing.assert_array_equal(raw[..., 2], swapped[..., 0])


def test_channel_alias_is_found(tmp_path):
    # "GFP" is an alias for FITC
    d = tmp_path / "S1"
    d.mkdir()
    img = np.full((8, 8), 7, dtype=np.uint16)
    tifffile.imwrite(d / "S1_raw_CH(GFP)_CH2.ome.tif", img)
    out = load_raw_tiff_sample("S1", tmp_path, ["FITC"])
    np.testing.assert_array_equal(out[..., 0], img)


def test_missing_channel_raises_with_helpful_message(tmp_path):
    (tmp_path / "S1").mkdir()
    with pytest.raises(FileNotFoundError, match="FITC"):
        load_raw_tiff_sample("S1", tmp_path, ["FITC"])


def _write_channels(root, name, channels, value=5):
    d = root / name
    d.mkdir()
    for ch in channels:
        tifffile.imwrite(d / f"{name}_{ch}.tif", np.full((8, 8), value, dtype=np.uint16))


def test_one_missing_channel_is_zero_filled_in_its_slot(tmp_path):
    _write_channels(tmp_path, "S1", ["DAPI", "FITC"])  # no TRITC
    raw = load_raw_tiff_sample("S1", tmp_path, ["TRITC", "FITC", "DAPI"])
    assert raw.shape == (8, 8, 3)
    assert raw.dtype == np.uint16
    assert (raw[..., 0] == 0).all()  # TRITC slot zero-filled, not shifted
    assert (raw[..., 1] == 5).all() and (raw[..., 2] == 5).all()


def test_two_missing_channels_still_raises(tmp_path):
    _write_channels(tmp_path, "S1", ["DAPI"])
    with pytest.raises(FileNotFoundError, match="TRITC"):
        load_raw_tiff_sample("S1", tmp_path, ["TRITC", "FITC", "DAPI"])


def test_get_missing_channels_and_is_segmentable(tmp_path):
    from src.io_utils import get_missing_channels, is_segmentable, missing_channels

    chans = ["TRITC", "FITC", "DAPI"]
    _write_channels(tmp_path, "two", ["DAPI", "FITC"])
    _write_channels(tmp_path, "one", ["DAPI"])
    _write_channels(tmp_path, "three", chans)

    assert get_missing_channels("two", tmp_path, chans) == ["TRITC"]
    assert get_missing_channels("three", tmp_path, chans) == []

    def files(n):
        return list((tmp_path / n).glob("*.tif"))

    assert is_segmentable(files("three"), chans)
    assert is_segmentable(files("two"), chans)
    assert not is_segmentable(files("one"), chans)
    assert missing_channels(files("one"), chans) == ["TRITC", "FITC"]


def test_load_sample_returns_bio7_resized_to_raw(sample_data_dir, tmp_path):
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    sem = np.zeros((128, 128), dtype=np.uint8)  # half the raw resolution
    sem[32:96, 32:96] = 1
    np.save(pred_dir / "Sample_01.npy", sem)

    raw, bio7 = load_sample("Sample_01", sample_data_dir, pred_dir, ["TRITC", "FITC", "DAPI"])
    assert raw.shape[:2] == (256, 256)
    assert all(m.shape == (256, 256) for m in bio7.values())
    assert bio7["Whole Root"].sum() == 128 * 128  # 64x64 block, doubled in each axis


def test_custom_channel_keyword_matches_and_defaults_still_work():
    from pathlib import Path

    from src.config import set_channel_aliases
    from src.io_utils import match_channel_file

    files = [Path("s1_cell_wall.tif"), Path("s1_FITC.tif"), Path("s1_TRITC.tif")]
    try:
        assert match_channel_file(files, "DAPI") is None
        set_channel_aliases({"DAPI": "cell_wall"})
        assert match_channel_file(files, "DAPI") == Path("s1_cell_wall.tif")
        assert match_channel_file(files, "FITC") == Path("s1_FITC.tif")
    finally:
        set_channel_aliases(None)
