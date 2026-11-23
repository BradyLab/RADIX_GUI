"""Keep the requirements*.txt files consistent with each other.

requirements.txt is the full environment. requirements-ci.txt is a slim subset of it
(no torch & co.) so CI stays fast, and the dev/typecheck files repeat a few tool pins.
Nothing used to stop those copies drifting, so a bump in one file could silently leave CI
testing different library versions than users install.

Rules enforced here:
  1. A package pinned (==) in more than one requirements file has the same version in all.
  2. Everything pinned in requirements-ci.txt is also pinned in requirements.txt (so
     the CI file really is a subset: a dependency removed from the real environment can't
     linger in CI).
"""

import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def _norm(name: str) -> str:
    """PEP 503 normalisation, so Pillow == pillow and PyYAML == pyyaml."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN.match(line.strip())  # comments, blanks and `-r other.txt` lines don't match
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


REQ_FILES = sorted(ROOT.glob("requirements*.txt"))


def test_requirements_files_are_found():
    names = {p.name for p in REQ_FILES}
    assert {"requirements.txt", "requirements-ci.txt"} <= names


def test_pin_parser_handles_real_world_lines(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text(
        "# comment\n\n-r other.txt\nPyYAML==6.0.3\nPillow == 12.2.0  # trailing\n"
        "uvicorn[standard]==0.50.2\nnumpy==1.26.4 ; python_version < '3.12'\npytest\nruff>=0.1\n"
    )
    assert _pins(f) == {
        "pyyaml": "6.0.3",
        "pillow": "12.2.0",
        "uvicorn": "0.50.2",
        "numpy": "1.26.4",
    }


def test_same_package_has_same_pinned_version_everywhere():
    seen = defaultdict(dict)  # package -> {file name: version}
    for path in REQ_FILES:
        for pkg, ver in _pins(path).items():
            seen[pkg][path.name] = ver

    conflicts = {pkg: files for pkg, files in seen.items() if len(set(files.values())) > 1}
    assert not conflicts, "version mismatch between requirements files:\n" + "\n".join(
        f"  {pkg}: {files}" for pkg, files in sorted(conflicts.items())
    )


def test_ci_requirements_are_a_subset_of_the_real_environment():
    full = _pins(ROOT / "requirements.txt")
    ci = _pins(ROOT / "requirements-ci.txt")
    assert ci, "requirements-ci.txt has no pinned packages?"
    missing = sorted(set(ci) - set(full))
    assert not missing, (
        f"pinned in requirements-ci.txt but not in requirements.txt: {missing}. "
        "Add them to requirements.txt, or remove them from the CI file."
    )


@pytest.mark.parametrize("path", REQ_FILES, ids=lambda p: p.name)
def test_every_requirements_file_is_readable(path):
    assert path.read_text(encoding="utf-8").strip()
