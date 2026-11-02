"""Non-learning baselines (P6). All return continuous scores so PR-AUC is a ranking metric.
Group rates are estimated on the training split only."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config


def cvss_score(df: pd.DataFrame) -> pd.Series:
    """cvss_at_tpred; missing ranks lowest (reported overall and on rows with CVSS)."""
    return df["cvss_at_tpred"].fillna(-1.0)


def cvss_high(df: pd.DataFrame, threshold: float = 7.0) -> pd.Series:
    """Common practice rule; report precision/recall, not PR-AUC."""
    return (df["cvss_at_tpred"] >= threshold).astype(int)


def group_rate(df: pd.DataFrame, split: pd.Series, col: str, label: str = "label_12m",
               smoothing: float = 20.0) -> pd.Series:
    """KEV rate per group learned on train, shrunk to the train base rate."""
    tr = df[(split == "train") & df[label].notna()]
    base = tr[label].mean()
    g = tr.groupby(tr[col].fillna("missing"))[label].agg(["sum", "count"])
    rate = (g["sum"] + smoothing * base) / (g["count"] + smoothing)
    return df[col].fillna("missing").map(rate).fillna(base)


def keyword_score(texts: pd.Series, rules=config.KEYWORD_RULE) -> pd.Series:
    pats = [re.compile(re.escape(r), re.I) for r in rules]
    t = texts.fillna("")
    # count of rules matched, small tie-breaker on length so the ranking is total
    return t.map(lambda s: sum(bool(p.search(s)) for p in pats)) + t.str.len() * 1e-6
