from pathlib import Path

import numpy as np
import tifffile as tiff

from src.config import get_channel_aliases
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


def load_raw_tiff_sample(sample_name, data_dir, channels):
    channel_imgs = []
    data_dir = Path(data_dir)

    sample_dir = data_dir / sample_name

    if sample_dir.exists():
        search_root = sample_dir
    else:
        matches = [p for p in data_dir.iterdir() if p.is_dir() and p.name == sample_name]
        search_root = matches[0] if matches else data_dir

    all_tifs = list(search_root.rglob("*.tif*"))
    for ch in channels:
        match = match_channel_file(all_tifs, ch)

        if match is None:
            raise FileNotFoundError(
                f"Could not find {ch} TIFF for sample '{sample_name}' under {search_root}. "
                f"Tried keywords: {get_channel_aliases(ch)}"
            )

        img = tiff.imread(match)
        channel_imgs.append(img)

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
