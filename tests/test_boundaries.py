import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _imports(folder, other):
    pat = re.compile(rf"^\s*(from|import)\s+{other}\b", re.M)
    return [str(f) for f in (ROOT / folder).rglob("*.py") if pat.search(f.read_text())]

def test_src_never_imports_submission():
    assert _imports("src", "submission") == []

def test_submission_never_imports_src():
    assert _imports("submission", "src") == []

def test_submission_never_reads_synthetic():
    hits = [str(f) for f in (ROOT / "submission").rglob("*.py")
            if "outputs_synthetic" in f.read_text()]
    assert hits == []
