"""The 2x2x2x2 leakage design: cells, inflation coefficients, main effects, interactions."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .labels import CLEAN, Protocol

FACTORS = ["random_split", "current_desc", "keep_exploit", "keep_zero_days"]   # A B C D


def all_cells() -> list[Protocol]:
    return [Protocol(*bits) for bits in itertools.product([False, True], repeat=4)]


def encoder_cells() -> list[Protocol]:
    """Clean, each single leaky factor, and all four leaky (6 cells)."""
    singles = [Protocol(**{f: True}) for f in FACTORS]
    return [CLEAN, *singles, Protocol(True, True, True, True)]


def inflation(leaky: dict, clean: dict) -> dict:
    """Ratios on PR-AUC and on lift (PR-AUC / base rate), which separates the leakage
    effect from a change in base rate (zero-days raise it)."""
    return {"pr_auc_ratio": leaky["pr_auc"] / clean["pr_auc"],
            "lift_ratio": leaky["lift"] / clean["lift"],
            "base_rate_ratio": leaky["base_rate"] / clean["base_rate"]}


def factorial_effects(results: pd.DataFrame, metric: str = "lift") -> pd.DataFrame:
    """results: one row per cell x seed with FACTORS (bool) and `metric`.
    Fits log(metric) ~ A*B*C*D (main + two-way) by least squares on the seed means.
    exp(main effect) = multiplicative inflation of that factor, averaged over the others."""
    g = results.groupby(FACTORS, as_index=False)[metric].mean()
    X = {f: g[f].astype(float) * 2 - 1 for f in FACTORS}           # effect coding (-1, +1)
    cols = {"intercept": np.ones(len(g)), **X}
    for a, b in itertools.combinations(FACTORS, 2):
        cols[f"{a}:{b}"] = X[a] * X[b]
    M = np.column_stack(list(cols.values()))
    coef, *_ = np.linalg.lstsq(M, np.log(g[metric].to_numpy()), rcond=None)
    out = pd.DataFrame({"term": list(cols), "coef": coef})
    # effect coding: +1 vs -1 differs by 2*coef on the log scale
    out["inflation_factor"] = np.where(out["term"] == "intercept", np.nan, np.exp(2 * out["coef"]))
    return out


def evaluate_both(model_fn, cell: Protocol, get_cell, clean_test) -> dict:
    """Dual evaluation: on the cell's own test rows (the number a leaky study would
    publish) and on the clean test rows."""
    Xtr, ytr, Xte, yte = get_cell(cell)
    score_own = model_fn(Xtr, ytr, Xte)
    Xc, yc = clean_test
    score_clean = model_fn(Xtr, ytr, Xc)
    return {"own": (yte, score_own), "clean": (yc, score_clean)}
