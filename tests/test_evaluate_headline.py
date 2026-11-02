"""Runs against a throwaway git repo so the real outputs/ is never touched."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "src", tmp_path / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    (tmp_path / "outputs" / "tables").mkdir(parents=True)
    (tmp_path / "outputs" / "tables" / "rq2.csv").write_text("a\n1\n")

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    from src import config, evaluate
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "OUTPUTS", tmp_path / "outputs")
    return tmp_path, evaluate


def test_untracked_outputs_do_not_count_as_dirty(repo):
    root, ev = repo
    (root / "outputs" / "tables" / "new.csv").write_text("b\n2\n")
    ev.write_headline_numbers({"max_inflation": (4.27, "outputs/tables/rq2.csv", "B")}, False)


def test_dirty_code_blocks(repo):
    root, ev = repo
    (root / "src" / "config.py").write_text("# edited\n")
    with pytest.raises(RuntimeError, match="Commit"):
        ev.write_headline_numbers({"x": (1.0, "outputs/tables/rq2.csv", "")}, False)


def test_synthetic_blocks(repo):
    _, ev = repo
    with pytest.raises(RuntimeError, match="Synthetic"):
        ev.write_headline_numbers({"x": (1.0, "outputs/tables/rq2.csv", "")}, True)


def test_two_calls_merge_and_provenance(repo):
    root, ev = repo
    ev.write_headline_numbers({"max_inflation": (4.27, "outputs/tables/rq2.csv", "B")}, False)
    ev.write_headline_numbers({"text_pr_auc": (0.042, "outputs/tables/rq2.csv", "clean")}, False)
    h = json.loads((root / "outputs" / "headline_numbers.json").read_text())
    assert h["values"] == {"max_inflation": 4.27, "text_pr_auc": 0.042}
    assert h["is_synthetic"] is False and h["zenodo_doi"] is None
    assert len(h["provenance"]["max_inflation"]["git_hash"]) == 40


def test_missing_source_table(repo):
    _, ev = repo
    with pytest.raises(FileNotFoundError):
        ev.write_headline_numbers({"x": (1.0, "outputs/tables/nope.csv", "")}, False)
