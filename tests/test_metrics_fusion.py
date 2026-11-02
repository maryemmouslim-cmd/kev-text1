import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config, evaluate, fusion, leakage, labels


def _toy(n=3000, seed=0, rate=0.01):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    p = np.clip(rng.beta(1, 50, n) + 0.08 * y * rng.random(n), 0, 1)
    q = np.clip(p + rng.normal(0, 0.01, n), 0, 1)
    return y, p, q


def test_precision_recall_effort():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.arange(10, 0, -1) / 10
    assert evaluate.precision_at_k(y, p, 3) == pytest.approx(2 / 3)
    assert evaluate.recall_at_k(y, p, 3) == pytest.approx(2 / 3)
    assert evaluate.effort_for_recall(y, p, 1.0) == 1.0
    assert evaluate.effort_for_recall(y, p, 0.5) == 0.3


def test_delong_matches_sklearn_auc():
    y, p, q = _toy()
    r = evaluate.delong(y, p, q)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, p)) and r["auc_b"] == pytest.approx(roc_auc_score(y, q))
    assert evaluate.delong(y, p, p)["p"] == 1.0


def test_paired_bootstrap_reproducible():
    y, p, q = _toy()
    a = evaluate.paired_bootstrap(y, p, q, n=300, seed=3)
    b = evaluate.paired_bootstrap(y, p, q, n=300, seed=3)
    assert a == b and a["lo"] <= a["delta"] <= a["hi"]


def test_mcc_at_half_is_useless_but_budget_is_not():
    y, p, _ = _toy()
    assert evaluate.mcc_at(y, p, 0.5) == 0.0
    assert evaluate.mcc_top_k(y, p) > 0


def test_ece_equal_mass_perfect_calibration():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 40, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert evaluate.ece_equal_mass(y, p) < 0.002
    assert evaluate.ece_equal_mass(y, np.clip(p * 3, 0, 1)) > 0.02


def test_metrics_keys():
    y, p, _ = _toy()
    m = evaluate.metrics(y, p, threshold=0.05)
    for k in ("pr_auc", "roc_auc", "lift", "precision@budget", "mcc@budget", "mcc@threshold", "brier", "ece"):
        assert k in m


def test_selection_lock_guards_test(sandbox):
    y, p, _ = _toy()
    with pytest.raises(RuntimeError, match="selection.lock"):
        evaluate.test_metrics("tfidf-lr", y, p)
    evaluate.lock_model_selection("tfidf-lr", 0.03, {"C": 1.0}, 0.04)
    assert "mcc@threshold" in evaluate.test_metrics("tfidf-lr", y, p)
    with pytest.raises(RuntimeError, match="exists"):
        evaluate.lock_model_selection("tfidf-lr", 0.05, {"C": 10.0}, 0.04)
    with pytest.raises(RuntimeError, match="not in selection"):
        evaluate.test_metrics("modernbert", y, p)


def test_best_encoder_lock(sandbox):
    evaluate.lock_model_selection("modernbert", 0.05, {}, 0.1)
    with pytest.raises(RuntimeError, match="lock these"):
        evaluate.lock_best_encoder(["modernbert", "deberta"])
    evaluate.lock_model_selection("deberta", 0.06, {}, 0.1)
    assert evaluate.lock_best_encoder(["modernbert", "deberta"]) == "deberta"
    with pytest.raises(RuntimeError, match="already"):
        evaluate.lock_best_encoder(["modernbert", "deberta"])


def test_best_by_validation():
    assert evaluate.best_by_validation({"a": 0.02, "b": 0.05, "c": 0.04}) == "b"


def test_run_count_table():
    t = evaluate.run_count_table()
    assert t.loc[t["category"] == "RQ2 factorial", "runs"].item() == 48
    assert t.loc[t["category"] == "RQ2 encoder", "runs"].item() == 18
    assert t.loc[t["category"] == "RQ3 encoders/QLoRA", "runs"].item() == 20


def test_log_run_and_budget(sandbox):
    with evaluate.log_run("tfidf-lr", 0, "----", is_synthetic=True) as rec:
        rec["metrics"] = {"pr_auc": 0.1}
    b = evaluate.compute_budget(is_synthetic=True)
    assert b["runs"].item() == 1


def test_mondrian_conformal_coverage():
    rng = np.random.default_rng(0)
    n = 40_000
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.03 + 0.25 * y + rng.normal(0, 0.08, n), 0.001, 0.999)
    cal, te = slice(0, n // 2), slice(n // 2, n)
    mc = fusion.MondrianConformal().fit(p[cal], y[cal], {0: 0.1, 1: 0.1})
    cov = mc.coverage(p[te], y[te])
    assert cov["coverage_0"] >= 0.88 and cov["coverage_1"] >= 0.88


def test_conformal_vacuous_with_few_positives():
    mc = fusion.MondrianConformal().fit(np.array([.1, .2, .9]), np.array([0, 0, 1]), {0: .1, 1: .1})
    assert mc.q[1] == np.inf


def test_calibrators_monotone():
    rng = np.random.default_rng(0)
    p = rng.random(5000); y = (rng.random(5000) < p ** 2).astype(int)
    for m in ("isotonic", "platt"):
        c = fusion.Calibrator(m).fit(p, y)
        out = c.transform(np.sort(p))
        assert np.all(np.diff(out) >= -1e-9)


def test_factorial_cells_and_effects():
    cells = leakage.all_cells()
    assert len(cells) == 16 and len({c.name for c in cells}) == 16
    assert leakage.encoder_cells()[0].is_clean and len(leakage.encoder_cells()) == 6
    rows = []
    for c in cells:   # true model: A doubles lift, B x1.5, others none
        lift = 10 * (2 if c.random_split else 1) * (1.5 if c.current_desc else 1)
        rows.append({**{f: getattr(c, f) for f in leakage.FACTORS}, "lift": lift})
    eff = leakage.factorial_effects(pd.DataFrame(rows)).set_index("term")
    assert eff.loc["random_split", "inflation_factor"] == pytest.approx(2.0)
    assert eff.loc["current_desc", "inflation_factor"] == pytest.approx(1.5)
    assert eff.loc["keep_exploit", "inflation_factor"] == pytest.approx(1.0)


def test_inflation_ratios():
    r = leakage.inflation({"pr_auc": .2, "lift": 10, "base_rate": .02}, {"pr_auc": .05, "lift": 5, "base_rate": .01})
    assert r == {"pr_auc_ratio": 4.0, "lift_ratio": 2.0, "base_rate_ratio": 2.0}
