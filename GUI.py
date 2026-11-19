import subprocess
import sys
import re

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from src.ui_widgets import range_slider_with_boxes

from src.io_utils import (
    load_sample,
    infer_species,
)

from src.intensity_quant import (
    BARRIER_TISSUES,
    normalize_channel,
    normalize_percentile_rgb,
    compute_threshold_percentile_suggestions,
    measure_barrier_intensity_batch,
    measure_root_morphometrics,
)

from src.plotting_utils import (
    to_uint8,
    crop_to_mask,
)


# -------------------------
# CONSTANTS
# -------------------------
CHANNELS = ["TRITC", "FITC", "DAPI"]
CH_IDX = {"TRITC": 0, "FITC": 1, "DAPI": 2}
OVERLAY_CHANNELS = ["FITC", "TRITC"]
OVERLAY_COLORS = {"FITC": [0, 255, 0], "TRITC": [255, 0, 0]}
LINE_COLORS = {"FITC": "#2ca02c", "TRITC": "#d62728"}
TISSUES = ["Epidermis", "Exodermis", "Endodermis", "Cortex", "Vascular"]

RAW = "Raw intensity"
NORM = "Normalized intensity"

# Always use prediction masks in the lab version
MASK_SOURCE = "Prediction (.npy)"


# -------------------------
# PAGE + SIDEBAR (one-time setup)
# -------------------------
st.set_page_config(layout="wide")

_paths_ok = (
    Path(st.session_state.get("data_dir", "data")).exists()
    and Path(st.session_state.get("model_dir", "model")).exists()
)

with st.sidebar.expander("Setup (folders)", expanded=not _paths_ok):
    data_dir_input = st.text_input(
        "Image folder",
        "data",
        key="data_dir",
        help=(
            "Folder containing one subfolder per sample, each with its "
            "TRITC/FITC/DAPI TIFF files. Defaults to the 'data' folder next "
            "to this app — copy your images there, or type a full path."
        ),
    )

    model_dir_input = st.text_input(
        "Model checkpoint file or folder",
        "model",
        key="model_dir",
        help=(
            "Defaults to the bundled 'model' folder next to this app "
            "(contains the trained RADIX checkpoint). Only change this if "
            "you were given a different checkpoint."
        ),
    )

    pred_dir_override = st.text_input(
        "Prediction folder (leave blank for auto)",
        "",
        key="pred_dir_override",
        help=(
            "By default predictions are saved to <image_folder>/../ml/predictions/. "
            "Set this only if your predictions are stored somewhere else."
        ),
    )

DATA_DIR = Path(data_dir_input)
EXP_DIR = DATA_DIR.parent
ML_DIR = EXP_DIR / "ml"
PRED_DIR = ML_DIR / "predictions"
MODEL_DIR = Path(model_dir_input)

if pred_dir_override.strip():
    PRED_DIR = Path(pred_dir_override.strip())

if not data_dir_input.strip() or not model_dir_input.strip():
    st.info("Please enter a data folder and model checkpoint/folder to begin.")
    st.stop()

st.sidebar.write("Data folder:", "✅" if DATA_DIR.exists() else "❌")
st.sidebar.write("Model path:", "✅" if MODEL_DIR.exists() else "❌")
st.sidebar.write("Predictions:", "✅" if PRED_DIR.exists() else "⚠️ run segmentation first")
st.sidebar.caption(f"Prediction folder: `{PRED_DIR}`")

st.title("RADIX")

samples = sorted(p.stem for p in PRED_DIR.glob("*.npy")) if PRED_DIR.exists() else []


# -------------------------
# HELPERS
# -------------------------
@st.cache_resource(show_spinner=False, max_entries=12)
def load_sample_cached(sample_name, data_dir, pred_dir):
    """Cached so moving a threshold slider doesn't re-read TIFFs from disk."""
    return load_sample(sample_name, Path(data_dir), Path(pred_dir), CHANNELS)


def _mode_suffix(mode):
    return "raw" if mode == RAW else "norm"


def _key(tissue, ch, mode):
    return f"{tissue.lower()}_{ch.lower()}_{_mode_suffix(mode)}"


def _default_range(mode):
    return (600, 65535) if mode == RAW else (15, 100)


def _bounds(mode):
    return (0, 65535) if mode == RAW else (0, 100)


def _thr_store():
    """Persistent threshold values (slider widgets only exist for the visible tissue)."""
    return st.session_state.setdefault("thr_store", {})


def _set_threshold(tissue, ch, mode, lo, hi):
    k = _key(tissue, ch, mode)
    _thr_store()[k] = (lo, hi)
    # Drop widget state so the slider re-seeds from the store on next render
    for suffix in ("_range", "_min", "_max"):
        st.session_state.pop(k + suffix, None)


def threshold_slider(tissue, ch, mode):
    k = _key(tissue, ch, mode)
    lo, hi = _thr_store().get(k, _default_range(mode))
    if f"{k}_range" not in st.session_state:
        st.session_state[f"{k}_range"] = (lo, hi)
        st.session_state[f"{k}_min"] = lo
        st.session_state[f"{k}_max"] = hi

    mn, mx = _bounds(mode)
    if mode == RAW:
        label, step = f"{ch} raw threshold", 10
    else:
        label, step = f"{ch} normalized threshold (%)", 1

    lo, hi = range_slider_with_boxes(label, mn, mx, _default_range(mode), step=step, key=k)
    _thr_store()[k] = (lo, hi)
    return lo, hi


def build_threshold_dict(mode):
    """{(tissue, channel): (lo, hi)} in the units the measurement code expects."""
    out = {}
    for tissue in BARRIER_TISSUES:
        for ch in OVERLAY_CHANNELS:
            lo, hi = _thr_store().get(_key(tissue, ch, mode), _default_range(mode))
            if mode == NORM:
                lo, hi = lo / 100, hi / 100
            out[(tissue, ch)] = (lo, hi)
    return out


# --- slider-button callbacks (run before widgets are re-created) ---
def cb_reset(tissue, mode):
    for ch in OVERLAY_CHANNELS:
        _set_threshold(tissue, ch, mode, *_default_range(mode))


def cb_copy_to_all(tissue, mode):
    for ch in OVERLAY_CHANNELS:
        lo, hi = _thr_store().get(_key(tissue, ch, mode), _default_range(mode))
        for t in BARRIER_TISSUES:
            _set_threshold(t, ch, mode, lo, hi)


def cb_apply_suggestion(pcol, scope, tissue):
    df = st.session_state.get("threshold_suggestions_df")
    if df is None or df.empty:
        return
    for _, row in df.iterrows():
        if scope == "tissue" and row["Tissue"] != tissue:
            continue
        lo = int(round(float(row[pcol]) / 10.0) * 10)
        lo = max(0, min(lo, 65535))
        _set_threshold(row["Tissue"], row["Channel"], RAW, lo, 65535)


def cb_step_sample(delta):
    cur = st.session_state.get("sample_selectbox")
    if cur in samples:
        i = samples.index(cur) + delta
        st.session_state["sample_selectbox"] = samples[max(0, min(i, len(samples) - 1))]


def tissue_mask(bio7, tissue):
    mask = bio7[tissue] > 0
    if tissue == "Cortex":
        mask = mask & ~(bio7["Aerenchyma"] > 0)
    return mask


def compute_views(raw, bio7, region, mode, thr, auto_crop):
    """Build preview images + threshold masks for one sample/tissue."""
    mask = tissue_mask(bio7, region)
    default = (0, 65535) if mode == RAW else (0.0, 1.0)

    rgb = to_uint8(normalize_percentile_rgb(raw))
    overlays, thr_only, ch_vals, pct = {}, {}, {}, {}

    for ch in OVERLAY_CHANNELS:
        img = raw[..., CH_IDX[ch]].astype(np.float32)
        vals = img if mode == RAW else normalize_channel(img)
        lo, hi = thr.get((region, ch), default)

        wall = mask & (vals >= lo) & (vals <= hi)
        ch_vals[ch] = vals[mask]
        n_mask = int(mask.sum())
        pct[ch] = (100.0 * int(wall.sum()) / n_mask) if n_mask else 0.0

        ov = rgb.copy()
        ov[wall] = OVERLAY_COLORS[ch]
        overlays[ch] = ov

        t = np.zeros_like(rgb)
        t[wall] = [255, 255, 255]
        thr_only[ch] = t

    if auto_crop and mask.any():
        _, (y0, y1, x0, x1) = crop_to_mask(rgb, mask)
        rgb = rgb[y0:y1, x0:x1]
        overlays = {c: v[y0:y1, x0:x1] for c, v in overlays.items()}
        thr_only = {c: v[y0:y1, x0:x1] for c, v in thr_only.items()}

    return {
        "rgb": rgb,
        "overlays": overlays,
        "thr_only": thr_only,
        "ch_vals": ch_vals,
        "pct": pct,
    }


def density_figure(vals, mode, ch, region, lo, hi, sugg_row=None):
    """KDE of tissue intensities with the current threshold range marked."""
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde

    vals = vals[np.isfinite(vals)]
    log_x = mode == RAW
    if log_x:
        vals = vals[vals > 0]
    if vals.size < 2 or vals.min() == vals.max():
        return None
    if vals.size > 50000:
        vals = np.random.default_rng(0).choice(vals, 50000, replace=False)

    try:
        kde = gaussian_kde(vals)
    except Exception:
        return None
    x_vals = np.linspace(vals.min(), vals.max(), 400)
    y_vals = kde(x_vals)

    is_dark = st.get_option("theme.base") == "dark"
    template = "plotly_dark" if is_dark else "plotly_white"
    ref_color = "rgba(255,255,255,0.6)" if is_dark else "rgba(0,0,0,0.6)"
    color = LINE_COLORS[ch]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", fill="tozeroy",
                             name="Density", line=dict(color=color)))

    def _vline(x, label, line_color, width, dash=None, y_label=1.0, size=11, angle=0):
        # Draw as a shape + separate annotation. add_vline(annotation_text=...)
        # is avoided: on a log axis Plotly puts annotations in log10 units but
        # shapes in data units, so labels land at 10^x and blow up the axis range.
        if not np.isfinite(x) or (log_x and x <= 0):
            return
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                      line=dict(color=line_color, width=width, dash=dash))
        fig.add_annotation(x=np.log10(x) if log_x else x, y=y_label, yref="paper",
                           text=label, showarrow=False, yanchor="bottom" if angle == 0 else "top",
                           font=dict(size=size, color=line_color), textangle=angle)

    if sugg_row is not None:
        for col in sugg_row.index:
            if col.startswith("p") and col[1:].isdigit():
                try:
                    _vline(float(sugg_row[col]), col, ref_color, 1, dash="dot",
                           y_label=0.98, size=9, angle=-90)
                except Exception:
                    pass

    for x, label in ((lo, "min"), (hi, "max")):
        _vline(x, label, color, 2)

    # Keep the axis on the data, so far-away lines (e.g. max=65535) can't stretch it
    xr = (np.log10(vals.min()), np.log10(vals.max())) if log_x else (vals.min(), vals.max())
    pad = 0.05 * (xr[1] - xr[0])
    fig.update_xaxes(range=[xr[0] - pad, xr[1] + pad])

    fig.update_layout(
        template=template,
        title=dict(text=f"{ch} — {region}", font=dict(size=13)),
        xaxis_title="Raw intensity" if log_x else "Normalized intensity",
        xaxis_type="log" if log_x else "linear",
        yaxis_title="Density",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


@st.fragment
def suggestions_panel(region, threshold_mode):
    """Percentile-suggestion controls.

    A fragment, so the slow all-samples scan only reruns this block — the
    preview and spot-check elsewhere on the page don't grey out while it runs.
    Anything that changes state used outside the fragment (the suggestions
    table feeding the density plots, the slider values) ends with a full-app
    rerun.
    """
    st.caption("Scans every sample to suggest raw-intensity thresholds per tissue/channel.")

    if st.button("Compute suggestions from all images"):
        st.session_state.pop("sugg_msg", None)
        with st.spinner("Scanning all samples for intensity statistics..."):
            suggestions_df = compute_threshold_percentile_suggestions(
                samples, DATA_DIR, PRED_DIR, CHANNELS,
                mask_source=MASK_SOURCE, gt_mask_dir=None,
            )
        errors = getattr(suggestions_df, "_errors", [])

        if suggestions_df.empty:
            st.session_state["sugg_msg"] = (
                "error",
                "No valid samples found. Make sure prediction .npy files "
                f"exist in: {PRED_DIR}",
                errors,
            )
        else:
            if errors:
                st.session_state["sugg_msg"] = (
                    "warning",
                    f"{len(errors)} sample(s) failed to load and were skipped.",
                    errors,
                )
            st.session_state["threshold_suggestions_df"] = suggestions_df
            st.rerun()  # full-app rerun so the density plots pick up the new lines

    msg = st.session_state.get("sugg_msg")
    if msg:
        kind, text, errors = msg
        (st.error if kind == "error" else st.warning)(text)
        if errors:
            with st.expander(f"Show errors ({len(errors)})"):
                for e in errors[:20]:
                    st.caption(e)

    sugg_df = st.session_state.get("threshold_suggestions_df")
    if sugg_df is None or sugg_df.empty:
        return

    st.dataframe(sugg_df, hide_index=True)

    pcols = [c for c in sugg_df.columns if c.startswith("p") and c[1:].isdigit()]
    if threshold_mode != RAW:
        st.caption("Switch to *Raw intensity* mode to apply suggestions.")
    elif pcols:
        pcol = st.selectbox(
            "Use as minimum threshold", pcols,
            index=min(2, len(pcols) - 1), key="apply_pcol",
        )
        a1, a2 = st.columns(2)
        # Plain buttons + st.rerun (not on_click): the sliders live outside this
        # fragment, so they only refresh on a full rerun.
        with a1:
            if st.button("Apply to this tissue", width="stretch"):
                cb_apply_suggestion(pcol, "tissue", region)
                st.rerun()
        with a2:
            if st.button("Apply to all", width="stretch"):
                cb_apply_suggestion(pcol, "all", region)
                st.rerun()


# -------------------------
# TABS
# -------------------------
tab_seg, tab_tune, tab_batch = st.tabs(
    ["1 · Segmentation", "2 · Threshold tuning", "3 · Batch & export"]
)


# =========================
# TAB 1: SEGMENTATION
# =========================
with tab_seg:
    st.markdown("### Run segmentation model")
    st.caption(
        "Runs predict.py on raw per-sample folders and saves full-size predictions "
        "for downstream RootLensX analysis."
    )

    flash = st.session_state.pop("seg_flash", None)
    if flash:
        st.success(flash)

    if samples:
        st.caption(f"{len(samples)} prediction file(s) already found in `{PRED_DIR}`.")
    else:
        st.warning("No predictions found yet — run segmentation to get started.")

    seg_input_dir = str(DATA_DIR)
    seg_model_dir = str(MODEL_DIR)
    seg_output_dir = str(ML_DIR)
    seg_pred_dir = str(PRED_DIR)

    st.caption(f"Raw image input: `{seg_input_dir}`")
    st.caption(f"Model checkpoint/folder: `{seg_model_dir}`")
    st.caption(f"Segmentation output: `{seg_output_dir}`")
    st.caption(f"Prediction masks: `{seg_pred_dir}`")

    seg_save_vis = st.checkbox(
        "Save segmentation preview PNGs",
        value=False,
        key="seg_save_vis",
    )

    run_segmentation = st.button("Run segmentation", type="primary")

    if run_segmentation:
        if not Path(seg_input_dir).exists():
            st.error(f"Input folder does not exist: {seg_input_dir}")
        elif not Path(seg_model_dir).exists():
            st.error(f"Model checkpoint/folder does not exist: {seg_model_dir}")
        else:
            cmd = [
                sys.executable,
                "-u",
                "predict.py",
                "--data-dir", seg_input_dir,
                "--model-dir", seg_model_dir,
                "--output-dir", seg_output_dir,
            ]

            if not seg_save_vis:
                cmd.append("--no-vis")

            with st.expander("Show command", expanded=False):
                st.code(" ".join(cmd))

            progress_bar = st.progress(0)
            progress_text = st.empty()
            log_box = st.empty()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            total = None
            done = 0
            logs = []
            current_sample = ""

            for line in process.stdout:
                line = line.rstrip()
                logs.append(line)

                if line.startswith("Discovered "):
                    m = re.search(r"Discovered\s+(\d+)\s+samples", line)
                    if m:
                        total = int(m.group(1))
                        progress_text.write(f"Progress: 0 / {total} (0.0%)")

                elif line.startswith("-- "):
                    current_sample = line.replace("-- ", "").strip()
                    done += 1
                    if total:
                        pct = done / total * 100
                        progress_bar.progress(min(done / total, 1.0))
                        progress_text.write(f"Progress: {done} / {total} ({pct:.1f}%)")
                    else:
                        progress_text.write(f"Processed: {done}")

                log_box.info(f"Current sample: {current_sample}" if current_sample else "Starting...")

            return_code = process.wait()

            if return_code == 0:
                load_sample_cached.clear()  # predictions changed on disk
                st.session_state["seg_flash"] = (
                    f"Segmentation finished. Predictions saved to {seg_pred_dir}. "
                    "Continue to the Threshold tuning tab."
                )
                st.rerun()
            else:
                st.error("Segmentation failed.")
                st.code("\n".join(logs) if logs else "(no output captured)")


# =========================
# TAB 2: THRESHOLD TUNING
# =========================
threshold_mode = st.session_state.get("threshold_mode", RAW)

with tab_tune:
    if not samples:
        st.info(
            f"No prediction files found in `{PRED_DIR}`. Run segmentation in the "
            "first tab, or check the prediction folder path in the sidebar."
        )
    else:
        if st.session_state.get("sample_selectbox") not in samples:
            st.session_state["sample_selectbox"] = samples[0]

        left, right = st.columns([1, 2.2], gap="large")

        # ---------- LEFT: all controls ----------
        with left:
            with st.container(border=True):
                st.markdown("##### Sample")
                c_prev, c_sel, c_next = st.columns([1, 5, 1])
                with c_prev:
                    st.button("◀", key="prev_sample", on_click=cb_step_sample, args=(-1,),
                              width="stretch")
                with c_next:
                    st.button("▶", key="next_sample", on_click=cb_step_sample, args=(1,),
                              width="stretch")
                with c_sel:
                    sample = st.selectbox(
                        "Sample", samples, key="sample_selectbox",
                        label_visibility="collapsed",
                    )
                st.caption(f"Sample {samples.index(sample) + 1} / {len(samples)}")

            with st.container(border=True):
                st.markdown("##### Tissue & thresholds")
                region = st.radio("Tissue", TISSUES, key="region_select", horizontal=True)

                threshold_mode = st.radio(
                    "Threshold mode",
                    [RAW, NORM],
                    key="threshold_mode",
                    horizontal=True,
                    help=(
                        "Positive-pixel masks are generated from either raw or normalized "
                        "images; intensities are always measured on raw data."
                    ),
                )

                for ch in OVERLAY_CHANNELS:
                    threshold_slider(region, ch, threshold_mode)

                b1, b2 = st.columns(2)
                with b1:
                    st.button("Reset tissue", on_click=cb_reset,
                              args=(region, threshold_mode), width="stretch")
                with b2:
                    st.button("Copy to all tissues", on_click=cb_copy_to_all,
                              args=(region, threshold_mode), width="stretch",
                              help="Apply this tissue's FITC/TRITC ranges to every tissue.")

                st.caption(
                    "Thresholds for other tissues are kept when you switch tissue — "
                    "see the summary in the *Batch & export* tab."
                )

            with st.expander("Percentile suggestions"):
                suggestions_panel(region, threshold_mode)

            with st.container(border=True):
                st.markdown("##### Random spot-check")
                n_spot = st.number_input(
                    "Number of random samples",
                    min_value=1, max_value=max(1, len(samples)),
                    value=min(4, max(1, len(samples))), step=1,
                )
                s1, s2 = st.columns(2)
                with s1:
                    if st.button("Draw samples", width="stretch"):
                        rng = np.random.default_rng()
                        st.session_state["spot_samples"] = rng.choice(
                            samples, size=min(int(n_spot), len(samples)), replace=False
                        ).tolist()
                with s2:
                    if st.button("Clear", width="stretch"):
                        st.session_state.pop("spot_samples", None)

        thr = build_threshold_dict(threshold_mode)

        # ---------- RIGHT: preview ----------
        with right:
            # View options: one row, one column per option, above the sample title
            v1, v2, v3 = st.columns(3)
            with v1:
                auto_crop = st.checkbox("Auto-crop to selected tissue", value=True,
                                        key="auto_crop_preview")
            with v2:
                show_masks = st.checkbox("Show threshold-only masks", value=False,
                                         key="show_threshold_only")
            with v3:
                show_hist = st.checkbox("Show intensity distributions", value=True,
                                        key="show_intensity_hist")

            try:
                raw, bio7 = load_sample_cached(sample, str(DATA_DIR), str(PRED_DIR))
                views = compute_views(raw, bio7, region, threshold_mode, thr, auto_crop)
            except Exception as e:
                st.error(f"Could not load {sample}: {e}")
                views = None

            if views is not None:
                st.subheader(f"{sample} — {region}")
                col_raw, col_fitc, col_tritc = st.columns(3)
                ch_cols = {"FITC": col_fitc, "TRITC": col_tritc}

                with col_raw:
                    st.image(views["rgb"], caption="Raw image", width="stretch")

                sugg_df = st.session_state.get("threshold_suggestions_df")

                for ch in OVERLAY_CHANNELS:
                    lo, hi = thr[(region, ch)]
                    with ch_cols[ch]:
                        st.image(views["overlays"][ch], caption=f"{ch} overlay",
                                 width="stretch")
                        st.caption(f"{views['pct'][ch]:.1f}% of {region} pixels above threshold")

                        if show_masks:
                            st.image(views["thr_only"][ch], caption=f"{ch} threshold only",
                                     width="stretch")

                        if show_hist:
                            sugg_row = None
                            if sugg_df is not None and not sugg_df.empty and threshold_mode == RAW:
                                match = sugg_df[(sugg_df["Tissue"] == region) &
                                                (sugg_df["Channel"] == ch)]
                                if len(match):
                                    sugg_row = match.iloc[0]

                            fig = density_figure(views["ch_vals"][ch], threshold_mode, ch,
                                                 region, lo, hi, sugg_row)
                            if fig is None:
                                st.warning(f"No usable {ch} pixels for {region}.")
                            else:
                                st.plotly_chart(fig, width="stretch", key=f"hist_{ch}")

        # ---------- FULL WIDTH: spot-check ----------
        spot = [s for s in st.session_state.get("spot_samples", []) if s in samples]
        if spot:
            st.divider()
            st.subheader(f"Random spot-check ({region}, current thresholds)")
            for name in spot:
                try:
                    raw_i, bio7_i = load_sample_cached(name, str(DATA_DIR), str(PRED_DIR))
                    v = compute_views(raw_i, bio7_i, region, threshold_mode, thr, auto_crop)
                except Exception as e:
                    st.warning(f"{name}: {e}")
                    continue

                st.markdown(f"**{name}**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(v["rgb"], caption="Raw", width="stretch")
                with c2:
                    st.image(v["overlays"]["FITC"],
                             caption=f"FITC overlay ({v['pct']['FITC']:.1f}%)", width="stretch")
                with c3:
                    st.image(v["overlays"]["TRITC"],
                             caption=f"TRITC overlay ({v['pct']['TRITC']:.1f}%)", width="stretch")


# =========================
# TAB 3: BATCH & EXPORT
# =========================
with tab_batch:
    st.markdown("### Barrier intensity quantification")
    st.caption(
        "Measures epidermis/exodermis/cortex/endodermis/vascular TRITC and FITC. "
        "Positive-pixel masks are generated from either raw or normalized images "
        "(depending on threshold mode); intensities are always measured on raw data."
    )

    if not samples:
        st.info("No predictions yet — run segmentation first.")
    else:
        batch_thr = build_threshold_dict(threshold_mode)

        st.markdown(f"**Thresholds that will be used** — mode: *{threshold_mode}*")
        scale = 100 if threshold_mode == NORM else 1
        unit = " (%)" if threshold_mode == NORM else ""
        summary = pd.DataFrame(
            [
                {
                    "Tissue": t,
                    **{f"{ch} min{unit}": round(batch_thr[(t, ch)][0] * scale) for ch in OVERLAY_CHANNELS},
                    **{f"{ch} max{unit}": round(batch_thr[(t, ch)][1] * scale) for ch in OVERLAY_CHANNELS},
                }
                for t in TISSUES
            ]
        )
        st.dataframe(summary, hide_index=True)
        st.caption("Adjust these in the Threshold tuning tab (mode and ranges are shared).")

        include_thresholds_output = st.checkbox(
            "Include thresholds in output",
            value=True,
            key="include_thresholds_output",
        )

        run_barrier_batch = st.button(f"Run barrier intensity batch ({len(samples)} samples)",
                                      type="primary")

        if run_barrier_batch:
            batch_rows = []
            progress = st.progress(0, text="Running barrier intensity batch...")

            for i, sample_name in enumerate(samples):
                try:
                    raw_i, bio7_i = load_sample(sample_name, DATA_DIR, PRED_DIR, CHANNELS)

                    rows = measure_barrier_intensity_batch(
                        raw_i,
                        bio7_i,
                        threshold_mode,
                        batch_thr,
                    )

                    rows["Sample"] = sample_name
                    rows["Species"] = infer_species(sample_name)
                    rows["Mask_source"] = MASK_SOURCE
                    # Root geometry + tissue areas, incl. aerenchyma (pixel units)
                    rows.update(measure_root_morphometrics(bio7_i))

                    if include_thresholds_output:
                        rows["Threshold_mode"] = threshold_mode

                        for (tissue_name, channel_name), (low_thr, high_thr) in batch_thr.items():
                            prefix = f"{tissue_name}_{channel_name}"
                            rows[f"{prefix}_threshold_min"] = low_thr
                            rows[f"{prefix}_threshold_max"] = high_thr

                    batch_rows.append(rows)

                except Exception as e:
                    batch_rows.append({
                        "Sample": sample_name,
                        "Species": infer_species(sample_name),
                        "Error": str(e),
                    })

                progress.progress((i + 1) / len(samples),
                                  text=f"Processed {i + 1} / {len(samples)}: {sample_name}")

            progress.empty()
            st.session_state["barrier_batch_df"] = pd.DataFrame(batch_rows)

        if "barrier_batch_df" in st.session_state:
            result_df = st.session_state["barrier_batch_df"]
            st.markdown("#### Results")

            if "Error" in result_df.columns:
                n_err = int(result_df["Error"].notna().sum())
                if n_err:
                    st.warning(f"{n_err} sample(s) failed — see the Error column.")

            st.dataframe(result_df)

            st.download_button(
                "Download barrier intensity CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="barrier_intensity_results.csv",
                mime="text/csv",
            )
