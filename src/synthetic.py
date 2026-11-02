"""Synthetic dataset with the same columns as dataset.parquet, for tests and dry runs
only. Every row carries is_synthetic=True; numbers from it mean nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd

VENDORS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Cyberdyne"]
SIGNAL = ["allows remote code execution", "allows unauthenticated attackers to bypass authentication",
          "leads to command injection"]
NOISE = ["causes a denial of service", "allows cross-site scripting", "discloses sensitive information",
         "may cause a crash in specific conditions"]


def make(n: int = 6000, seed: int = 0, start="2022-01-01", end="2025-06-30",
         kev_snapshot="2026-10-16") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    t0 = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    vendor = rng.choice(VENDORS, n)
    sig = rng.random(n) < 0.15
    phrase = np.where(sig, rng.choice(SIGNAL, n), rng.choice(NOISE, n))
    prod = np.char.add(vendor.astype(str), " Server")
    ver = [f"{a}.{b}.{c}" for a, b, c in rng.integers(0, 12, (n, 3))]
    desc = [f"A flaw in {p} {v} {ph}." for p, v, ph in zip(prod, ver, phrase)]
    # positives: signal + Acme/Globex vendor bias + noise
    logit = -5.5 + 2.8 * sig + 1.0 * np.isin(vendor, ["Acme", "Globex"]) + rng.normal(0, .5, n)
    pos = rng.random(n) < 1 / (1 + np.exp(-logit))
    lag = np.where(rng.random(n) < 0.15, rng.integers(-3, 7, n), rng.integers(8, 365, n))
    kev = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    kev[pos] = (t0 + pd.to_timedelta(lag, unit="D"))[pos]
    zero = pos & (lag <= 7)
    desc_cur = [d + (" This vulnerability is actively exploited in the wild." if z or (p and rng.random() < .3)
                     else "") for d, z, p in zip(desc, zero, pos)]
    desc_orig = [d + (" Microsoft is aware of limited, targeted attacks." if z else "")
                 for d, z in zip(desc, zero)]
    epss = np.clip(rng.beta(1, 30, n) + 0.08 * pos + 0.03 * sig, 0, 1)
    cvss = np.round(np.clip(rng.normal(6.5, 1.6, n) + 1.0 * sig, 0, 10), 1)
    cvss_missing = rng.random(n) < np.where(t0 >= pd.Timestamp("2024-03-01"), 0.5, 0.1)
    df = pd.DataFrame({
        "cve_id": [f"CVE-{t.year}-{100000 + i}" for i, t in enumerate(t0)],
        "state": "PUBLISHED", "published_date": t0.normalize(),
        "t_pred": t0.normalize() + pd.Timedelta(days=1), "lang_ok": True,
        "desc_original": desc_orig, "desc_current": desc_cur,
        "vendors": [[v] for v in vendor], "products": [[p] for p in prod], "vendor": vendor,
        "cwe": rng.choice(["CWE-79", "CWE-89", "CWE-787", "CWE-22", "CWE-78", None], n),
        "cvss_at_tpred": np.where(cvss_missing, np.nan, cvss),
        "cvss_missing_at_tpred": cvss_missing,
        "cvss_source": np.where(cvss_missing, "missing", "cna"),
        "epss_at_pred": epss, "epss_lag_days": rng.choice([0, 0, 0, 1, 2, 9], n),
        "epss_model_version": np.where(t0 >= pd.Timestamp("2025-03-17"), "v4", "v3"),
        "kev_date_added": kev.to_numpy(), "nvd_published": t0.normalize(),
        "is_synthetic": True,
    })
    miss = rng.random(n) < 0.05
    df.loc[miss, "desc_original"] = None
    from .labels import add_labels
    return add_labels(df, pd.Timestamp(kev_snapshot))
