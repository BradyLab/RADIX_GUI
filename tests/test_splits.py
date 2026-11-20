import pytest

from src import splits


class _S:
    def __init__(self, species):
        self.species = species


_SAMPLES = {
    str(splits.TRAIN_DIR): [_S("Rice"), _S("Tomato"), _S("Millet")],
    str(splits.VAL_DIR): [_S("Sorghum")],
    str(splits.TEST_DIR): [_S("Tomato"), _S("Rice")],
    str(splits.ONESHOT_DIR): [_S("Rice"), _S("Tomato")],
}


class _FakeRegistry:
    """Stands in for SampleRegistry so no real data directory is needed."""

    def __init__(self, data_dir, require_annotations=True):
        self.samples = _SAMPLES[str(data_dir)]


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch):
    monkeypatch.setattr(splits, "SampleRegistry", _FakeRegistry)


def _species(split):
    return {k: sorted(s.species for s in v) for k, v in split.items()}


def test_strategy_a_keeps_everything():
    assert _species(splits.get_split("A")) == {
        "train": ["Millet", "Rice", "Tomato"],
        "val": ["Sorghum"],
        "test": ["Rice", "Tomato"],
    }


def test_monocot_and_dicot_strategies_partition_species():
    assert _species(splits.get_split("B-mono")) == {
        "train": ["Millet", "Rice"],
        "val": ["Sorghum"],
        "test": ["Rice"],
    }
    assert _species(splits.get_split("B-dico")) == {
        "train": ["Tomato"],
        "val": [],
        "test": ["Tomato"],
    }


def test_single_species_overrides_strategy():
    out = _species(splits.get_split("B-dico", species="Rice"))
    assert out == {"train": ["Rice"], "val": [], "test": ["Rice"]}


def test_oneshot_puts_everything_in_test():
    out = splits.get_split("oneshot")
    assert out["train"] == [] and out["val"] == []
    assert len(out["test"]) == 2
    assert _species(splits.get_split("oneshot", species="Tomato"))["test"] == ["Tomato"]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        splits.get_split("Z")


def test_print_split_summary_runs(capsys):
    splits.print_split_summary({"train": [], "val": [], "test": []})
    assert "TRAIN: 0 samples" in capsys.readouterr().out
