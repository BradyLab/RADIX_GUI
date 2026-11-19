"""Real-browser end-to-end tests (Playwright driving the running Streamlit app)."""
import re

import pytest
from playwright.sync_api import Page, expect

from conftest import SAMPLE_DATA_DIR, requires_sample_data

LOAD_TIMEOUT = 60_000


def open_app(page: Page, app_url: str, data_dir: str | None = SAMPLE_DATA_DIR):
    page.goto(app_url)
    expect(page.get_by_role("heading", name="RADIX")).to_be_visible(timeout=LOAD_TIMEOUT)
    if data_dir:
        box = page.get_by_label("Image folder", exact=True)
        if not box.is_visible():  # sidebar "Setup" expander may be collapsed
            page.get_by_text("Setup (folders)").click()
        box.fill(data_dir)
        box.press("Enter")


def open_tab(page: Page, name: str):
    page.get_by_role("tab", name=re.compile(name)).click()


def test_tabs_render(page: Page, app_url):
    open_app(page, app_url, data_dir=None)
    for name in ("Segmentation", "Threshold tuning", "Batch & export"):
        expect(page.get_by_role("tab", name=re.compile(name))).to_be_visible()


def test_missing_folder_gives_guidance(page: Page, app_url):
    open_app(page, app_url, data_dir="does_not_exist_xyz")
    open_tab(page, "Threshold tuning")
    expect(page.get_by_text("No prediction files found")).to_be_visible(timeout=LOAD_TIMEOUT)


@requires_sample_data
def test_threshold_tuning_workflow(page: Page, app_url):
    open_app(page, app_url)
    open_tab(page, "Threshold tuning")

    # Preview, per-channel overlays and density plots appear
    expect(page.get_by_text("FITC overlay", exact=True).first).to_be_visible(timeout=LOAD_TIMEOUT)
    expect(page.get_by_text("TRITC overlay", exact=True).first).to_be_visible()
    expect(page.locator(".js-plotly-plot").first).to_be_visible(timeout=LOAD_TIMEOUT)

    # Only the selected tissue's two range sliders (FITC, TRITC) are shown;
    # each range slider exposes a min and a max thumb.
    expect(page.get_by_role("slider")).to_have_count(4)

    # Sample navigation
    expect(page.get_by_text(re.compile(r"Sample 1 / \d+"))).to_be_visible()
    page.get_by_role("button", name="▶").click()
    expect(page.get_by_text(re.compile(r"Sample 2 / \d+"))).to_be_visible(timeout=LOAD_TIMEOUT)
    page.get_by_role("button", name="◀").click()
    expect(page.get_by_text(re.compile(r"Sample 1 / \d+"))).to_be_visible(timeout=LOAD_TIMEOUT)

    # Switch tissue: heading follows and the preview re-renders
    page.get_by_text("Cortex", exact=True).first.click()
    expect(page.get_by_role("heading", name=re.compile("— Cortex"))).to_be_visible(timeout=LOAD_TIMEOUT)

    # Threshold-only masks toggle
    page.get_by_text("Show threshold-only masks").click()
    expect(page.get_by_text("FITC threshold only")).to_be_visible(timeout=LOAD_TIMEOUT)

    # Normalized mode switches slider labels
    page.get_by_text("Normalized intensity", exact=True).first.click()
    expect(page.get_by_text("FITC normalized threshold (%)")).to_be_visible(timeout=LOAD_TIMEOUT)


@requires_sample_data
def test_random_spot_check(page: Page, app_url):
    open_app(page, app_url)
    open_tab(page, "Threshold tuning")
    # The results heading includes "(<tissue>, current thresholds)"; the
    # control panel's own "Random spot-check" heading does not.
    results = page.get_by_role("heading", name=re.compile(r"Random spot-check \("))
    expect(results).to_have_count(0)
    page.get_by_role("button", name="Draw samples").click()
    expect(results).to_be_visible(timeout=LOAD_TIMEOUT)
    page.get_by_role("button", name="Clear").click()
    expect(results).to_have_count(0, timeout=LOAD_TIMEOUT)


@requires_sample_data
def test_batch_tab_shows_threshold_summary(page: Page, app_url):
    open_app(page, app_url)
    open_tab(page, "Batch & export")
    expect(page.get_by_text("Thresholds that will be used")).to_be_visible(timeout=LOAD_TIMEOUT)
    expect(page.get_by_role("button", name=re.compile("Run barrier intensity batch"))).to_be_visible()


@pytest.mark.slow
@requires_sample_data
def test_run_batch_and_download(page: Page, app_url):
    open_app(page, app_url)
    open_tab(page, "Batch & export")
    page.get_by_role("button", name=re.compile("Run barrier intensity batch")).click()
    download_btn = page.get_by_role("button", name="Download barrier intensity CSV")
    expect(download_btn).to_be_visible(timeout=600_000)
    with page.expect_download() as dl:
        download_btn.click()
    assert dl.value.suggested_filename == "barrier_intensity_results.csv"
