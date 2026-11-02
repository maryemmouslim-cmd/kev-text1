import pytest


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every file the pipeline writes into tmp_path."""
    from src import config
    for name, rel in {"SPLITS_LOCK": "splits.lock.json", "SELECTION_LOCK": "selection.lock.json",
                      "LOCK_HISTORY": "lock_history.jsonl", "MANIFEST": "data/manifest.json",
                      "OUTPUTS": "outputs", "OUTPUTS_SYNTHETIC": "outputs_synthetic"}.items():
        monkeypatch.setattr(config, name, tmp_path / rel)
    return tmp_path


@pytest.fixture
def synth():
    from src import synthetic
    return synthetic.make(4000, seed=0)
