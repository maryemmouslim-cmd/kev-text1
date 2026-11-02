"""Data-quality report (Table 2), temporal profile, vendor/CWE reports, near duplicates,
selection-bias tests. assert_ready stops the pipeline on a failed essential check."""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .io_utils import verify_manifest
from .labels import assign_split


# ---------------------------------------------------------------- helpers
def holm(pvals: pd.Series) -> pd.Series:
    p = pvals.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.to_numpy())])
    return pd.Series(np.minimum(adj, 1.0), index=p.index).reindex(pvals.index)


def wilson(k, n, z: float = 1.959964):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def _contingency_test(a, b, c, d):
    """Fisher exact, or chi-square when every expected cell >= 5."""
    table = np.array([[a, b], [c, d]])
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / max(table.sum(), 1)
    if (exp >= 5).all():
        return "chi2", stats.chi2_contingency(table)[1]
    return "fisher", stats.fisher_exact(table)[1]


# ---------------------------------------------------------------- individual reports
def state_report(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("state", dropna=False).agg(
        n=("cve_id", "size"), n_kev=("kev_date_added", lambda s: s.notna().sum()))


def language_report(df: pd.DataFrame, col: str = "desc_original") -> pd.DataFrame:
    """No English description -> excluded. Low ASCII ratio -> flagged for manual review only."""
    t = df[col].fillna("")
    ascii_ratio = t.map(lambda s: sum(ch.isascii() for ch in s) / len(s) if s else 1.0)
    return pd.DataFrame({"no_english_description": ~df["lang_ok"].fillna(False),
                         "low_ascii_flag": ascii_ratio < 0.8}, index=df.index)


def restoration_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Is desc_original missing more often for positives? Overall and per year, Holm-adjusted."""
    d = df[df["label_12m"].notna()].assign(
        missing=df["desc_original"].isna(), year=df["published_date"].dt.year)
    rows = []
    for key, g in [("all", d)] + list(d.groupby("year")):
        a = int((g["missing"] & (g["label_12m"] == 1)).sum())
        b = int((~g["missing"] & (g["label_12m"] == 1)).sum())
        c = int((g["missing"] & (g["label_12m"] == 0)).sum())
        e = int((~g["missing"] & (g["label_12m"] == 0)).sum())
        test, p = _contingency_test(a, b, c, e)
        rows.append({"group": key, "miss_pos": a, "n_pos": a + b, "miss_neg": c, "n_neg": c + e,
                     "miss_rate_pos": a / max(a + b, 1), "miss_rate_neg": c / max(c + e, 1),
                     "test": test, "p": p})
    r = pd.DataFrame(rows).set_index("group")
    years = r.index != "all"
    r.loc[years, "p_holm"] = holm(r.loc[years, "p"])
    return r


def temporal_profile(df: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    d = df[df["label_12m"].notna()].assign(year=df["published_date"].dt.year,
                                           split=assign_split(df["published_date"]))
    g = d.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "n_pos": g["label_12m"].sum(),
        "pos_rate": g["label_12m"].mean(),
        "zero_day_rate": g["is_zero_day"].mean(),
        "desc_original_missing": g["desc_original"].apply(lambda s: s.isna().mean()),
        "cvss_missing_at_tpred": g["cvss_missing_at_tpred"].mean(),
        "desc_len_median": g["desc_original"].apply(lambda s: s.str.len().median()),
    })
    if "cwe" in d:
        out["top_cwe_share"] = g["cwe"].apply(
            lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    if "epss_model_version" in d:
        out["epss_v4_share"] = g["epss_model_version"].apply(lambda s: (s == "v4").mean())
    return out


def group_report(df: pd.DataFrame, col: str, min_n: int = 30) -> pd.DataFrame:
    """KEV rate per vendor or CWE, training split only, with Wilson intervals on the lift."""
    d = df[df["label_12m"].notna()]
    tr = d[assign_split(d["published_date"]) == "train"].copy()
    counts = tr[col].value_counts()
    tr[col] = tr[col].where(tr[col].map(counts).fillna(0) >= min_n, "other").fillna("missing")
    base = tr["label_12m"].mean()
    r = tr.groupby(col)["label_12m"].agg(n="size", n_pos="sum", pos_rate="mean")
    lo, hi = wilson(r["n_pos"], r["n"])
    r["lift"], r["lift_lo"], r["lift_hi"] = r["pos_rate"] / base, lo / base, hi / base
    return r.sort_values("n", ascending=False)


def length_report(texts: pd.Series, labels: pd.Series, tokenizer=None,
                  cutoffs=(256, 512)) -> pd.DataFrame:
    from .features import token_lengths
    n = token_lengths(texts, tokenizer)
    rows = {}
    for lab, g in n.groupby(labels):
        rows[f"label={int(lab)}"] = {"median": g.median(), "p95": g.quantile(.95),
                                    **{f"truncated@{c}": (g > c).mean() for c in cutoffs}}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- near duplicates (MinHash)
def _shingles(text: str, k: int = 3) -> set[str]:
    w = re.findall(r"\w+", (text or "").lower())
    return {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))} if w else set()


def minhash(texts: pd.Series, num_perm: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = (1 << 61) - 1
    a = rng.integers(1, p, num_perm, dtype=np.uint64)
    b = rng.integers(0, p, num_perm, dtype=np.uint64)
    sig = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, t in enumerate(texts):
        hs = np.array([zlib.crc32(s.encode()) for s in _shingles(t)], dtype=np.uint64)
        if len(hs):
            sig[i] = ((np.outer(hs, a) + b) % np.uint64(p)).min(axis=0)
    return sig


def near_duplicate_report(texts: pd.Series, split: pd.Series, threshold: float = 0.9,
                          bands: int = 16, query: str = "test", ref: str = "train") -> dict:
    """Share of `query` texts with a near duplicate (estimated Jaccard >= threshold) in `ref`.
    Rows are reported, never removed: duplicates are part of what RQ2 measures."""
    sig = minhash(texts)
    rows = sig.shape[1] // bands
    idx = np.arange(len(texts))
    is_q, is_r = (split == query).to_numpy(), (split == ref).to_numpy()
    buckets: dict = {}
    for i in idx[is_r]:
        for bnd in range(bands):
            buckets.setdefault((bnd, sig[i, bnd * rows:(bnd + 1) * rows].tobytes()), []).append(i)
    hits = []
    for i in idx[is_q]:
        cands = {j for bnd in range(bands)
                 for j in buckets.get((bnd, sig[i, bnd * rows:(bnd + 1) * rows].tobytes()), [])}
        best = max(((sig[i] == sig[j]).mean() for j in cands), default=0.0)
        hits.append(best >= threshold)
    return {"query": query, "ref": ref, "n_query": int(is_q.sum()),
            "share_with_near_duplicate": float(np.mean(hits)) if hits else 0.0}


def embedding_duplicate_share(emb_query: np.ndarray, emb_ref: np.ndarray, threshold: float) -> float:
    """Second view on duplicates. `threshold` is fixed after a manual audit of 100 pairs
    (train/val only) and locked; normalised embeddings expected."""
    best = np.max(emb_query @ emb_ref.T, axis=1) if len(emb_ref) else np.zeros(len(emb_query))
    return float(np.mean(best >= threshold))


# ---------------------------------------------------------------- full report + gate
def qc_report(df: pd.DataFrame) -> dict:
    pub = df["published_date"]
    rep = {
        "n_rows": len(df),
        "duplicate_cve_id": int(df["cve_id"].duplicated().sum()),
        "state": state_report(df),
        "rejected": int((df["state"] == "REJECTED").sum()),
        "no_english_description": int((~df["lang_ok"].fillna(False)).sum()),
        "desc_original_restored": float(df["desc_original"].notna().mean()),
        "desc_original_missing_by_label": df.groupby(df["label_12m"])["desc_original"]
                                            .apply(lambda s: int(s.isna().sum())).to_dict(),
        "nvd_vs_cve_date_diff_days": (df["nvd_published"].dt.normalize() - pub).dt.days.describe()
                                     if "nvd_published" in df else None,
        "kev_before_publication": int(df.get("kev_before_publication", pd.Series(False)).sum()),
        "censored": int(df["label_12m"].isna().sum()),
        "by_year_split": df.groupby([pub.dt.year, assign_split(pub)], dropna=False)
                           .size().rename("n"),
        "raw_files_changed": verify_manifest(),
        "cvss_missing_at_tpred": df.groupby(pub.dt.year)["cvss_missing_at_tpred"].mean(),
    }
    if df["label_12m"].notna().any():
        rep["restoration_bias"] = restoration_bias(df)
        rep["temporal_profile"] = temporal_profile(df)
    return rep


def assert_ready(rep: dict) -> None:
    errors = []
    if rep["duplicate_cve_id"]:
        errors.append(f"{rep['duplicate_cve_id']} duplicate cve_id")
    if rep["raw_files_changed"]:
        errors.append(f"raw files changed after download: {rep['raw_files_changed']}")
    if rep["desc_original_restored"] < 0.5:
        errors.append(f"desc_original restored for only {rep['desc_original_restored']:.0%}")
    if errors:
        raise RuntimeError("QC failed: " + "; ".join(errors))
