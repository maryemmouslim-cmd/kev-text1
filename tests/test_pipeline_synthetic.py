"""End-to-end dry run on synthetic data: lock -> clean TF-IDF/LR -> selection lock ->
fusion on val_a -> calibration + Conformal on val_b -> one test evaluation -> factorial
-> figures to outputs_synthetic/. Numbers mean nothing; the wiring is what is tested."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import config, evaluate, features, fusion, labels, leakage, models, plots


def test_end_to_end(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "synthetic dry run")
    X, y, split = labels.get_xy(synth)
    res = models.tfidf_lr(X, y, split, seed=0)
    s = res["scores"]
    va = average_precision_score(y[split == "val_a"], s["val_a"])
    thr = evaluate.best_mcc_threshold(y[split == "val_a"].to_numpy(), s["val_a"].to_numpy())
    evaluate.lock_model_selection("tfidf-lr", va, {"C": 1.0}, thr)

    # RQ4 fusion: fitted on val_a, calibrated on val_b, evaluated once on comparable test rows
    test_idx, counts = fusion.comparison_rows(synth, s["test"].index)
    assert counts["lag_exceeded"] > 0                      # synthetic has lag 9 rows
    fz = fusion.Fusion("text+epss").fit(s["val_a"], y[s["val_a"].index], synth)
    cal = fusion.Calibrator("isotonic").fit(fz.predict(s["val_b"], synth), y[s["val_b"].index])
    p_test = cal.transform(fz.predict(s["test"].loc[test_idx], synth))
    m = evaluate.test_metrics("tfidf-lr", y[test_idx], p_test)
    assert 0 <= m["pr_auc"] <= 1 and "brier" in m

    alpha = config.conformal_alpha(int(y[split == "val_b"].sum()))
    mc = fusion.MondrianConformal().fit(fz.predict(s["val_b"], synth), y[s["val_b"].index], alpha)
    cov = mc.coverage(fz.predict(s["test"].loc[test_idx], synth), y[test_idx])
    assert 0 <= cov["coverage_0"] <= 1

    fig = plots.reliability_diagram({"fused": evaluate.reliability_table(y[test_idx], p_test)})
    path = plots.save_figure(fig, "reliability", synth, is_synthetic=True)
    assert "outputs_synthetic" in path
    fig = plots.reliability_diagram({"fused": evaluate.reliability_table(y[test_idx], p_test)})
    with pytest.raises(RuntimeError, match="synthetic"):
        plots.save_figure(fig, "reliability", synth, is_synthetic=False)


def test_factorial_detects_planted_leaks(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "synthetic dry run")
    rows = []
    for cell in leakage.all_cells():
        X, y, split = labels.get_xy(synth, cell, seed=0)
        s = models.tfidf_lr(X, y, split, seed=0)["scores"]["test"]
        yt = y[s.index]
        m = evaluate.metrics(yt, s)
        rows.append({**{f: getattr(cell, f) for f in leakage.FACTORS}, "lift": m["lift"]})
    eff = leakage.factorial_effects(pd.DataFrame(rows)).set_index("term")
    # the synthetic data plants exploitation phrases in desc_current (B) and in zero-day
    # desc_original (C, D); the clean cell must not see them
    assert eff.loc["current_desc", "inflation_factor"] > 1.2
    fig = plots.inflation_bars(eff.reset_index())
    plots.save_figure(fig, "inflation", synth, is_synthetic=True)


def test_entity_masking_s7(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "synthetic dry run")
    X, y, split = labels.get_xy(synth)
    Xm = features.mask_frame(X, synth)
    assert not Xm["text"].str.contains("Acme").any() and Xm["text"].str.contains("VENDOR|PRODUCT").all()
