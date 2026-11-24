import numpy as np

from src.blob_counting import estimate_lumen

SCALE = {"cell_scale_px": 20.0}


def _two_cells(wall_thickness=2):
    """A 60x120 tissue block split into two equal compartments by a vertical wall."""
    tissue = np.zeros((100, 160), dtype=bool)
    tissue[20:80, 20:140] = True
    wall = np.zeros_like(tissue)
    mid = 80
    wall[:, mid - wall_thickness // 2 : mid + wall_thickness // 2] = True
    return tissue, wall


def test_wall_splits_tissue_into_two_lumens():
    tissue, wall = _two_cells()
    labels = estimate_lumen(tissue, wall, min_area=50, max_area=10_000, scale=SCALE)
    assert labels.shape == tissue.shape
    assert set(np.unique(labels)) == {0, 1, 2}
    assert not (labels > 0)[wall].any()  # nothing labelled inside the wall


def test_labels_stay_inside_tissue_and_are_consecutive():
    tissue, wall = _two_cells()
    labels = estimate_lumen(tissue, wall, min_area=50, scale=SCALE)
    assert not (labels > 0)[~tissue].any()
    ids = np.unique(labels[labels > 0])
    assert ids.tolist() == list(range(1, len(ids) + 1))


def test_min_area_filters_small_compartments():
    tissue, wall = _two_cells()
    labels = estimate_lumen(tissue, wall, min_area=10**6, scale=SCALE)
    assert not labels.any()


def test_works_without_explicit_scale():
    tissue, wall = _two_cells()
    labels = estimate_lumen(tissue, wall)  # derives scale from the tissue mask itself
    assert labels.shape == tissue.shape
