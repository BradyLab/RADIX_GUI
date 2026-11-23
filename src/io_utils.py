from pathlib import Path

import numpy as np
import tifffile as tiff

from src.mask_utils import resize_bio7_to_raw
from src.model_classes import unet_semantic_to_bio7

_SPECIES_KEYWORDS = [
    ("sorghum", "Sorghum"),
    ("rice", "Rice"),
    ("millet", "Millet"),
    ("tomato", "Tomato"),
]

CHANNEL_ALIASES = {
    "FITC": ["FITC", "GFP"],
    "TRITC": ["TRITC", "RFP"],
    "DAPI": ["DAPI", "CW", "CALCOFLUOR"],
}


def _safe_child(base_dir, name):
    """Join `name` onto `base_dir`, refusing anything that escapes it.

    Sample names are a single folder/file stem, so separators, drive letters
    and '..' have no legitimate use and are rejected.
    """
    name = str(name)
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid sample name: {name!r}")
    base = Path(base_dir).resolve()
    child = (base / name).resolve()
    if base != child and base not in child.parents:
        raise ValueError(f"Sample name escapes its folder: {name!r}")
    return child


def load_raw_tiff_sample(sample_name, data_dir, channels):
    channel_imgs = []
    data_dir = Path(data_dir).resolve()

    sample_dir = _safe_child(data_dir, sample_name)

    if sample_dir.exists():
        search_root = sample_dir
    else:
        matches = [p for p in data_dir.iterdir() if p.is_dir() and p.name == sample_name]
        search_root = matches[0] if matches else data_dir

    for ch in channels:
        ch_variants = CHANNEL_ALIASES.get(ch.upper(), [ch])

        matches = sorted(
            [
                p
                for p in search_root.rglob("*.tif*")
                if any(
                    (
                        f"_{alias.lower()}." in p.name.lower()
                        or f"_{alias.lower()}_" in p.name.lower()
                        or f"({alias.lower()})" in p.name.lower()
                        or f"-{alias.lower()}-" in p.name.lower()
                        or f"-{alias.lower()}." in p.name.lower()
                    )
                    for alias in ch_variants
                )
            ]
        )

        if not matches:
            raise FileNotFoundError(
                f"Could not find {ch} TIFF for sample '{sample_name}' under {search_root}. "
                f"Tried aliases: {ch_variants}"
            )

        img = tiff.imread(matches[0])
        channel_imgs.append(img)

    return np.stack(channel_imgs, axis=-1)


def load_sample(sample_name, data_dir, pred_dir, channels):
    raw = load_raw_tiff_sample(sample_name, data_dir, channels)

    pred_dir = Path(pred_dir).resolve()
    sem = np.load(_safe_child(pred_dir, f"{sample_name}.npy"), allow_pickle=False)

    bio7 = unet_semantic_to_bio7(sem, *sem.shape)
    bio7 = resize_bio7_to_raw(bio7, raw.shape)

    return raw, bio7


def infer_species(sample_name):
    name = sample_name.lower()
    for keyword, label in _SPECIES_KEYWORDS:
        if keyword in name:
            return label
    return "Unknown"
