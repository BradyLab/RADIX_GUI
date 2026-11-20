"""Import every heavy dependency and project module the slim CI environment skips.

Run from the repo root by the install-smoke workflow (after installing the real
requirements.txt). Exits non-zero, naming each failure, if anything won't import.
"""

import importlib
import sys

sys.path.insert(0, ".")  # repo root, so `src.*` resolves

MODULES = [
    # third-party
    "torch",
    "torchvision",
    "cellpose",
    "ultralytics",
    "timm",
    "albumentations",
    "pytorch_lightning",
    "segmentation_models_pytorch",
    "streamlit",
    # project modules that need the above
    "src.lightning_module",
    "src.unet_dataset",
    "src.augmentation",
]

failures = []
for name in MODULES:
    try:
        mod = importlib.import_module(name)
        print(f"ok    {name} {getattr(mod, '__version__', '')}".rstrip())
    except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        failures.append(name)

if failures:
    sys.exit(f"{len(failures)} import(s) failed: {', '.join(failures)}")
