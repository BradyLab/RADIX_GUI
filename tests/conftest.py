import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import tifffile

APP_DIR = Path(__file__).resolve().parent.parent
N_SAMPLES = 3


def _make_prediction(size: int = 256) -> np.ndarray:
    """Concentric-ring semantic mask using the 7 U-Net class labels.

    0=bg 1=epidermis 2=aerenchyma 3=endodermis 4=vascular 5=exodermis 6=cortex
    """
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2) / size
    sem = np.zeros((size, size), dtype=np.uint8)
    sem[r < 0.32] = 1   # epidermis (outermost ring)
    sem[r < 0.28] = 5   # exodermis
    sem[r < 0.24] = 6   # cortex
    sem[r < 0.14] = 3   # endodermis
    sem[r < 0.10] = 4   # vascular
    sem[(r > 0.18) & (r < 0.22) & (xx > size / 2 + 10)] = 2  # a little aerenchyma
    return sem


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> str:
    """Tiny synthetic dataset laid out like real data, so tests need no lab images.

        <tmp>/exp/Sample_NN/Sample_NN_raw_CH(<ch>)_CH<n>.ome.tif   (raw images)
        <tmp>/ml/predictions/Sample_NN.npy                          (segmentation)

    Returns the image folder (what you'd type into the app's "Image folder" box).
    """
    root = tmp_path_factory.mktemp("radix_data")
    data_dir = root / "exp"
    pred_dir = root / "ml" / "predictions"
    pred_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    for i in range(1, N_SAMPLES + 1):
        name = f"Sample_{i:02d}"
        sample_dir = data_dir / name
        sample_dir.mkdir(parents=True)
        for n, ch in enumerate(["DAPI", "FITC", "TRITC"], start=1):
            img = (rng.gamma(2.0, 2500.0, (256, 256)) + 100).clip(0, 65535).astype(np.uint16)
            tifffile.imwrite(sample_dir / f"{name}_raw_CH({ch})_CH{n}.ome.tif", img)
        np.save(pred_dir / f"{name}.npy", _make_prediction())

    return str(data_dir)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def app_url():
    """Start the Streamlit app once for the whole test session."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "GUI.py",
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=APP_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy")
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Streamlit did not start within 60s")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
