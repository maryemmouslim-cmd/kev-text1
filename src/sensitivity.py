"""Sensitivity analyses S1-S8 (Table 5.5) and the KEV retroactive curve (Figure 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import config


class NoStablePeriod(Exception):
    pass


def kev_retroactive_monthly(kev: pd.DataFrame, cves: pd.DataFrame) -> pd.DataFrame:
    """kev: cve_id + kev_date_added (one row per CVE, e.g. data.kev_union output);
    cves: cve_id + published_date. Fraction of monthly additions published > 365 days earlier."""
    k = (kev.groupby("cve_id", as_index=False)["kev_date_added"].min()
            .merge(cves[["cve_id", "published_date"]], on="cve_id", how="left"))
    k["retro"] = (k["kev_date_added"] - k["published_date"]).dt.days > 365
    month = k["kev_date_added"].dt.to_period("M")
    m = k.groupby(month).agg(n=("retro", "size"), n_retro=("retro", "sum"),
                             unmatched=("published_date", lambda s: int(s.isna().sum())))
    m["frac"] = m["n_retro"] / m["n"]
    return m


def stability_start(m: pd.DataFrame) -> pd.Period:
    m = m[m["n"] >= config.STABILITY_MIN_N]
    below = (m["frac"] < config.STABILITY_THRESHOLD)[::-1].cummin()[::-1]
    cands = below[below].index
    if len(cands) == 0:
        raise NoStablePeriod(f"max frac {m['frac'].max():.3f} >= {config.STABILITY_THRESHOLD}")
    return cands.min()


def s1_record(m: pd.DataFrame, df: pd.DataFrame, split: pd.Series) -> dict:
    """Written to splits.lock.json under 's1' in notebook 02."""
    try:
        start, rule = stability_start(m).to_timestamp(), "threshold"
    except NoStablePeriod:
        start, rule = pd.Timestamp(config.S1_FALLBACK_START), "fallback"
    tr = df[(split == "train") & df["label_12m"].notna()]
    kept = tr[tr["published_date"] >= start]
    t0 = config.TRAIN_START
    return {"start": str(start.date()), "rule": rule, "threshold": config.STABILITY_THRESHOLD,
            "months_dropped": max(0, (start.year - t0.year) * 12 + start.month - t0.month),
            "n_train_kept": int(len(kept)), "n_train_total": int(len(tr)),
            "n_pos_kept": int(kept["label_12m"].sum()), "n_pos_total": int(tr["label_12m"].sum())}


def embargo_mask(df: pd.DataFrame, split: pd.Series) -> pd.Series:
    """S8: keep train rows whose 12-month label was observable before the test period
    started (t0 + horizon < test start). Without this, train labels use KEV additions
    made during the test period."""
    test_start = pd.Timestamp(config.SPLITS["test"][0])
    late = (split == "train") & (df["published_date"] + pd.Timedelta(days=config.EMBARGO_DAYS) >= test_start)
    return ~late


def rank_agreement(clean: pd.Series, alt: pd.Series) -> dict:
    common = clean.index.intersection(alt.index)
    tau, _ = kendalltau(clean[common], alt[common])
    return {"kendall_tau": float(tau), "n_models": int(len(common))}


def summary_table(res: dict[str, dict]) -> pd.DataFrame:
    """Table 5.5. res[code] has pr_auc, ci_lo, ci_hi, delta, delta_lo, delta_hi (paired
    bootstrap on shared rows), per_model (Series of PR-AUC), n_pos. Analyses that change
    the label (S2, S3, S6) report ranking only."""
    clean, rows = res["clean"], []
    nan = float("nan")
    for code, r in res.items():
        if code == "clean":
            continue
        ok = code in config.COMPARABLE
        rows.append({"analysis": code, "description": config.SENSITIVITY_PLAN.get(code, ""),
                     "pr_auc": r["pr_auc"] if ok else nan,
                     "pr_auc_lo": r["ci_lo"] if ok else nan, "pr_auc_hi": r["ci_hi"] if ok else nan,
                     "delta_vs_clean": r["delta"] if ok else nan,
                     "delta_lo": r["delta_lo"] if ok else nan, "delta_hi": r["delta_hi"] if ok else nan,
                     **rank_agreement(clean["per_model"], r["per_model"]), "n_pos": r["n_pos"]})
    return pd.DataFrame(rows)
