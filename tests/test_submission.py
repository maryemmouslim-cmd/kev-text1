import json
import pytest
from submission import checks, journal, cover_letter

FULL = {"max_inflation": 4.27, "max_inflation_lo": 3.9, "max_inflation_hi": 4.7,
        "text_pr_auc": 0.042, "epss_pr_auc": 0.038}

def _write(p, values, synthetic=False, doi=None):
    p.write_text(json.dumps({"is_synthetic": synthetic, "values": values, "zenodo_doi": doi}))

@pytest.fixture
def headline(tmp_path, monkeypatch):
    p = tmp_path / "h.json"
    monkeypatch.setattr(checks, "HEADLINE", p)
    return p

# --- from the review ---
def test_highlight_too_long():
    with pytest.raises(ValueError, match="chars"):
        checks.check_highlights(["x" * 86, "ok", "ok"])

def test_abstract_match(headline):
    _write(headline, {"text_pr_auc": 0.04213, "x": 1.0})
    checks.check_abstract("We get 0.042 PR-AUC.")
    with pytest.raises(ValueError):
        checks.check_abstract("We get 0.051 PR-AUC.")

def test_headline_missing_flag(headline):
    headline.write_text(json.dumps({"values": {"x": 1.0}}))
    with pytest.raises(ValueError, match="synthetic"):
        checks.load_headline()

def test_mark_done_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "MANUAL_LOG", tmp_path / "done.jsonl")
    journal.mark_done("guide_for_authors_read", "checked 2026-10-16")
    assert "guide_for_authors_read" in journal.done_keys()

def test_cover_letter_pending_doi(headline):
    _write(headline, FULL, doi="PENDING")
    with pytest.raises(ValueError, match="zenodo_doi"):
        cover_letter.draft()

# --- added ---
def test_highlight_count():
    with pytest.raises(ValueError, match="highlights"):
        checks.check_highlights(["a", "b"])
    checks.check_highlights(["x" * 85, "b", "c"])          # 85 exactly is allowed

def test_headline_synthetic_true(headline):
    _write(headline, {"x": 1.0}, synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        checks.load_headline()

def test_headline_nan(headline):
    headline.write_text('{"is_synthetic": false, "values": {"x": NaN}}')
    with pytest.raises(ValueError, match="Missing"):
        checks.load_headline()

def test_abstract_times_rounding(headline):
    _write(headline, {"a": 8.4})
    checks.check_abstract("Random splits inflate PR-AUC by 8×.")
    _write(headline, {"a": 8.6})
    with pytest.raises(ValueError):
        checks.check_abstract("Random splits inflate PR-AUC by 8×.")

def test_abstract_percent(headline):
    _write(headline, {"a": 0.4213})
    checks.check_abstract("Recall drops by 42.1%.")

def test_abstract_ignores_plain_integers(headline):
    _write(headline, {"a": 0.042})
    checks.check_abstract("Over 12 months in 2022, PR-AUC was 0.042.")
    with pytest.raises(ValueError, match="no numerical"):
        checks.check_abstract("We study 12 months of 2022 data.")

def test_abstract_f1_decimal(headline):
    _write(headline, {"f1": 1.0})
    checks.check_abstract("F1 = 1.00 on the toy set.")

def test_mark_done_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "MANUAL_LOG", tmp_path / "done.jsonl")
    with pytest.raises(KeyError):
        journal.mark_done("not_an_item", "x")
    with pytest.raises(ValueError):
        journal.mark_done("word_limit_verified", "  ")
    journal.mark_done("doi_manual:10.1016/j.jisa.2025.104125", "opened on ScienceDirect")
    assert "doi_manual:10.1016/j.jisa.2025.104125" in journal.done_keys()

def test_cover_letter_ok(headline):
    _write(headline, FULL, doi="10.5281/zenodo.1234567")
    txt = cover_letter.draft()
    assert "4.3x" in txt and "0.042" in txt and "zenodo.1234567" in txt

def test_cover_letter_missing_value(headline):
    v = dict(FULL); del v["epss_pr_auc"]
    _write(headline, v, doi="10.5281/zenodo.1234567")
    with pytest.raises(ValueError, match="epss_pr_auc"):
        cover_letter.draft()

BIB_OK = """@article{lpass,
  title = {LPASS},
  journal = {Journal of Information Security and Applications},
  year = {2025}, volume = {93},
  doi = {10.1016/j.jisa.2025.104125}
}
@misc{kev, title = {CISA KEV catalog}, howpublished = {cisa.gov}}
"""
BIB_BAD = """@article{wrong,
  title = {Decision-oriented},
  journal = {Journal of Information Security and Applications},
  year = {2025},
  doi = {10.1016/j.cose.2025.100000}
}
"""

def test_references_ok(tmp_path, monkeypatch):
    p = tmp_path / "r.bib"; p.write_text(BIB_OK)
    monkeypatch.setattr(checks, "BIB", p)
    assert checks.check_references(online=False) == []

def test_references_wrong_journal(tmp_path, monkeypatch):
    p = tmp_path / "r.bib"; p.write_text(BIB_BAD)
    monkeypatch.setattr(checks, "BIB", p)
    with pytest.raises(ValueError, match="computers & security"):
        checks.check_references(online=False)

def test_references_no_doi(tmp_path, monkeypatch):
    p = tmp_path / "r.bib"
    p.write_text("@article{x, title={T}, journal={J}, year={2024}}\n")
    monkeypatch.setattr(checks, "BIB", p)
    with pytest.raises(ValueError, match="no DOI"):
        checks.check_references(online=False)

def test_assert_ready_unreachable_doi(tmp_path, monkeypatch, headline):
    _write(headline, FULL, doi="10.5281/zenodo.1234567")
    p = tmp_path / "r.bib"; p.write_text(BIB_OK)
    monkeypatch.setattr(checks, "BIB", p)
    monkeypatch.setattr(journal, "MANUAL_LOG", tmp_path / "done.jsonl")
    monkeypatch.setattr(checks, "_crossref", lambda e, doi: ([], True))   # 429 / network down
    for k in journal.MANUAL_ITEMS:
        journal.mark_done(k, "done")
    hl, ab = ["a", "b", "c"], "PR-AUC 0.042."
    with pytest.raises(ValueError, match="10.1016/j.jisa.2025.104125"):
        checks.assert_ready(hl, ab)
    journal.mark_done("doi_manual:10.1016/j.jisa.2025.104125", "opened manually")
    checks.assert_ready(hl, ab)                       # now passes

def test_assert_ready_404_blocks(tmp_path, monkeypatch, headline):
    _write(headline, FULL, doi="10.5281/zenodo.1234567")
    p = tmp_path / "r.bib"; p.write_text(BIB_OK)
    monkeypatch.setattr(checks, "BIB", p)
    monkeypatch.setattr(journal, "MANUAL_LOG", tmp_path / "done.jsonl")
    monkeypatch.setattr(checks, "_crossref",
                        lambda e, doi: ([f"{e['ID']}: DOI not found on Crossref"], False))
    for k in journal.MANUAL_ITEMS:
        journal.mark_done(k, "done")
    journal.mark_done("doi_manual:10.1016/j.jisa.2025.104125", "trying to bypass")
    with pytest.raises(ValueError, match="not found"):
        checks.assert_ready(["a", "b", "c"], "PR-AUC 0.042.")
