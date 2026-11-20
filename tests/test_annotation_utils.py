import numpy as np

from src.annotation_utils import parse_yolo_annotations, polygon_to_mask


def test_parse_yolo_scales_normalised_coords(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n3 0.5 0.5 0.6 0.5 0.6 0.6\n")
    anns = parse_yolo_annotations(f, img_w=200, img_h=100)
    assert [a["class_id"] for a in anns] == [0, 3]
    np.testing.assert_allclose(anns[0]["polygon"][0], [20, 10])
    np.testing.assert_allclose(anns[0]["polygon"][2], [180, 90])
    assert anns[0]["polygon"].dtype == np.float32


def test_parse_yolo_skips_degenerate_and_blank_lines(tmp_path):
    f = tmp_path / "a.txt"
    # first line has only 2 points, so it is dropped
    f.write_text("0 0.1 0.1 0.2 0.2\n\n1 0.1 0.1 0.5 0.1 0.5 0.5\n")
    anns = parse_yolo_annotations(f, 100, 100)
    assert [a["class_id"] for a in anns] == [1]


def test_polygon_to_mask_fills_rectangle():
    poly = np.array([[10, 10], [30, 10], [30, 20], [10, 20]], dtype=np.float32)
    mask = polygon_to_mask(poly, h=40, w=40)
    assert mask.shape == (40, 40) and mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}
    assert mask[15, 20] == 1 and mask[0, 0] == 0
    # inclusive polygon edges: area is close to the 20x10 rectangle
    assert 200 <= mask.sum() <= 250


def test_polygon_to_mask_fills_self_intersecting_bowtie_solid():
    # even-odd fill would leave holes; the function re-fills from the outer contour
    poly = np.array([[5, 5], [35, 35], [35, 5], [5, 35]], dtype=np.float32)
    mask = polygon_to_mask(poly, 40, 40)
    assert mask[20, 20] == 1
