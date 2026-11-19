"""Fast, browser-free tests using Streamlit's AppTest harness."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, N_SAMPLES


@pytest.fixture
def at(monkeypatch, sample_data_dir):
    monkeypatch.chdir(APP_DIR)  # the app uses relative paths ("data", "model")
    app = AppTest.from_file(str(APP_DIR / "GUI.py"), default_timeout=300)
    app.session_state["data_dir"] = sample_data_dir
    return app.run()


def test_no_predictions_shows_guidance(monkeypatch, tmp_path):
    monkeypatch.chdir(APP_DIR)
    app = AppTest.from_file(str(APP_DIR / "GUI.py"), default_timeout=60)
    app.session_state["data_dir"] = str(tmp_path / "empty")
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "1 · Segmentation", "2 · Threshold tuning", "3 · Batch & export",
    ]
    assert any("No prediction files found" in i.value for i in app.info)


def test_loads_without_errors(at):
    assert not at.exception
    assert not at.error
    assert [s.label for s in at.slider] == ["FITC raw threshold", "TRITC raw threshold"]
    assert len(at.selectbox(key="sample_selectbox").options) == N_SAMPLES


def test_thresholds_persist_across_tissue_switch(at):
    at.radio(key="region_select").set_value("Cortex").run()
    at.slider(key="cortex_fitc_raw_range").set_value((2000, 30000)).run()
    at.radio(key="region_select").set_value("Vascular").run()
    at.radio(key="region_select").set_value("Cortex").run()
    assert not at.exception
    assert at.session_state["thr_store"]["cortex_fitc_raw"] == (2000, 30000)
    assert at.slider(key="cortex_fitc_raw_range").value == (2000, 30000)


def test_copy_to_all_and_reset(at):
    at.slider(key="epidermis_fitc_raw_range").set_value((1500, 40000)).run()
    [b for b in at.button if b.label == "Copy to all tissues"][0].click().run()
    store = at.session_state["thr_store"]
    assert all(store[f"{t}_fitc_raw"] == (1500, 40000)
               for t in ["epidermis", "exodermis", "endodermis", "cortex", "vascular"])

    [b for b in at.button if b.label == "Reset tissue"][0].click().run()
    assert at.session_state["thr_store"]["epidermis_fitc_raw"] == (600, 65535)
    # Other tissues keep the copied value
    assert at.session_state["thr_store"]["cortex_fitc_raw"] == (1500, 40000)


def test_mode_switch_and_sample_navigation(at):
    at.radio(key="threshold_mode").set_value("Normalized intensity").run()
    assert not at.exception
    assert at.slider(key="epidermis_fitc_norm_range").value == (15, 100)

    first = at.session_state["sample_selectbox"]
    at.button(key="next_sample").click().run()
    assert at.session_state["sample_selectbox"] != first
    at.button(key="prev_sample").click().run()
    assert at.session_state["sample_selectbox"] == first


def test_percentile_suggestions_apply(at):
    """Scan runs in a fragment; applying a suggestion updates the sliders."""
    [b for b in at.button if b.label == "Compute suggestions from all images"][0].click().run()
    assert not at.exception
    df = at.session_state["threshold_suggestions_df"]
    assert not df.empty
    # N_samples counts samples (not pixels)
    assert (df["N_samples"] == N_SAMPLES).all()
    assert (df["N_samples"] < df["N_pixels"]).all()

    [b for b in at.button if b.label == "Apply to all"][0].click().run()
    assert not at.exception
    store = at.session_state["thr_store"]
    lo, hi = store["epidermis_fitc_raw"]
    assert lo != 600 and hi == 65535
    assert at.slider(key="epidermis_fitc_raw_range").value == (lo, hi)
