import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
# Folder whose parent contains ml/predictions (matches the bundled sample data).
SAMPLE_DATA_DIR = "data/Experiment_1_Control"


def _have_sample_data() -> bool:
    return (
        (APP_DIR / SAMPLE_DATA_DIR).exists()
        and any((APP_DIR / "data" / "ml" / "predictions").glob("*.npy"))
    )


requires_sample_data = pytest.mark.skipif(
    not _have_sample_data(),
    reason="needs data/Experiment_1_Control and data/ml/predictions/*.npy",
)


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
