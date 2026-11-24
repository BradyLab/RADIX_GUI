from pathlib import Path

import numpy as np
import tifffile as tiff

from src.config import MAX_MISSING_CHANNELS, get_channel_aliases
from src.mask_utils import resize_bio7_to_raw
from src.model_classes import unet_semantic_to_bio7

_SPECIES_KEYWORDS = [
    ("sorghum", "Sorghum"),
    ("rice", "Rice"),
    ("millet", "Millet"),
    ("tomato", "Tomato"),
]


def match_channel_file(files, channel):
    """Return the first file (sorted) whose name carries a keyword for ``channel``.

    Keywords come from config.get_channel_aliases (user-defined first, then
    defaults). Returns None if nothing matches.
    """
    files = sorted(files)
    for alias in get_channel_aliases(channel):
        a = alias.lower()
        tokens = (f"_{a}.", f"_{a}_", f"({a})", f"-{a}-", f"-{a}.")
        for f in files:
            name = Path(f).name.lower()
            if any(tok in name for tok in tokens):
                return f
    return None


def missing_channels(files, channels):
    """Channels in ``channels`` that have no matching file in ``files``."""
    return [ch for ch in channels if match_channel_file(files, ch) is None]


def is_segmentable(files, channels):
    """True if enough channels are present for the dropout-trained model.

    At most MAX_MISSING_CHANNELS may be absent (and never all of them).
    """
    n_missing = len(missing_channels(files, channels))
    return n_missing <= min(MAX_MISSING_CHANNELS, len(channels) - 1)


def _sample_search_root(sample_name, data_dir):
    data_dir = Path(data_dir)
    sample_dir = data_dir / sample_name

    if sample_dir.exists():
        return sample_dir
    matches = [p for p in data_dir.iterdir() if p.is_dir() and p.name == sample_name]
    return matches[0] if matches else data_dir


def get_missing_channels(sample_name, data_dir, channels):
    """Channels with no TIFF for this sample (these get zero-filled on load)."""
    search_root = _sample_search_root(sample_name, data_dir)
    return missing_channels(list(search_root.rglob("*.tif*")), channels)


def load_raw_tiff_sample(sample_name, data_dir, channels):
    """Stack channel TIFFs as (H, W, C) in ``channels`` order.

    A channel with no file is filled with zeros (what the model saw during
    channel-dropout training), as long as is_segmentable() allows it.
    """
    search_root = _sample_search_root(sample_name, data_dir)
    all_tifs = list(search_root.rglob("*.tif*"))

    if not is_segmentable(all_tifs, channels):
        absent = missing_channels(all_tifs, channels)
        raise FileNotFoundError(
            f"Sample '{sample_name}' under {search_root} is missing {absent}; "
            f"at most {MAX_MISSING_CHANNELS} channel(s) may be missing. "
            f"Tried keywords: { {ch: get_channel_aliases(ch) for ch in absent} }"
        )

    channel_imgs = []
    for ch in channels:
        match = match_channel_file(all_tifs, ch)
        channel_imgs.append(None if match is None else tiff.imread(match))

    ref = next(img for img in channel_imgs if img is not None)
    channel_imgs = [np.zeros_like(ref) if img is None else img for img in channel_imgs]

    return np.stack(channel_imgs, axis=-1)


def load_sample(sample_name, data_dir, pred_dir, channels):
    raw = load_raw_tiff_sample(sample_name, data_dir, channels)

    pred_dir = Path(pred_dir)
    sem = np.load(pred_dir / f"{sample_name}.npy")

    bio7 = unet_semantic_to_bio7(sem, *sem.shape)
    bio7 = resize_bio7_to_raw(bio7, raw.shape)

    return raw, bio7


def infer_species(sample_name):
    name = sample_name.lower()
    for keyword, label in _SPECIES_KEYWORDS:
        if keyword in name:
            return label
    return "Unknown"
