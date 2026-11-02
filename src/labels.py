"""Labels, right censoring, zero-days, exploit-phrase removal, temporal split with lock,
and the only sanctioned way to build X and y (get_xy)."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config
from .io_utils import (append_lock_history, atomic_write_json, git_state, manifest_event,
                       now_utc, read_json, sha256_obj)

# ===================================================================== labels
def assert_after_kev_launch(df: pd.DataFrame) -> None:
    bad = df["published_date"] < config.KEV_LAUNCH
    if bad.any():
        raise ValueError(f"{int(bad.sum())} CVEs published before KEV existed")


def is_zero_day(kev_date_added: pd.Series, t0: pd.Series, days: int = config.ZERO_DAY_DAYS) -> pd.Series:
    """KEV add - t0 <= days, negative differences (added before publication) included."""
    delta = (kev_date_added - t0).dt.days
    return (delta <= days).fillna(False).astype(int)


def add_labels(df: pd.DataFrame, kev_snapshot_date: pd.Timestamp,
               horizon: int = config.LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """label_12m, followup_complete, is_zero_day, label_any_kev, followup_days.

    A CVE whose horizon window has not ended strictly before the KEV snapshot date is
    right-censored: excluded (label NaN), never labelled 0 - positives included, so the
    positive rate is not inflated for recent CVEs.
    """
    out = df.copy()
    snap = pd.Timestamp(kev_snapshot_date).normalize()
    t0 = out["published_date"]
    delta = (out["kev_date_added"] - t0).dt.days
    out["followup_days"] = (snap - t0).dt.days
    out["followup_complete"] = (t0 + pd.Timedelta(days=horizon)) < snap
    pos = (delta <= horizon).fillna(False)
    out["label_12m"] = np.where(out["followup_complete"], pos.astype(int), np.nan)
    out["label_any_kev"] = out["kev_date_added"].notna().astype(int)
    out["is_zero_day"] = is_zero_day(out["kev_date_added"], t0)
    out["kev_before_publication"] = (delta < 0).fillna(False)
    return out


def relabel_horizon(df: pd.DataFrame, horizon: int, kev_snapshot_date) -> pd.Series:
    """S3: label for another horizon (6 or 24 months) with its own censoring."""
    return add_labels(df, kev_snapshot_date, horizon)["label_12m"]


# ===================================================================== text: exploit phrases
_PHRASE = re.compile("|".join(f"(?:{p})" for p in config.EXPLOIT_PHRASES), re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def has_exploit_phrase(text) -> int:
    return int(bool(text) and isinstance(text, str) and bool(_PHRASE.search(text)))


def remove_exploit_sentences(text):
    """Drop the whole sentence that contains an exploitation-status phrase."""
    if not isinstance(text, str) or not text:
        return text
    kept = [s for s in _SENT_SPLIT.split(text.strip()) if not _PHRASE.search(s)]
    return " ".join(kept)


# ===================================================================== temporal split + lock
def _bounds() -> dict:
    return {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in config.SPLITS.items()}


def assign_split(t0: pd.Series) -> pd.Series:
    s = pd.Series(pd.NA, index=t0.index, dtype="object")
    for name, (a, b) in _bounds().items():
        s[(t0 >= a) & (t0 <= b)] = name
    return s


def lock_hash() -> str:
    return sha256_obj(config.lockable())


LOCK_REASONS = {"initial_seal", "data_bug_fix", "source_refresh"}


def write_lock(df: pd.DataFrame, reason: str, note: str, extra: dict | None = None) -> dict:
    """Notebook 02: freeze boundaries + everything in config.lockable(), with counts."""
    if reason not in LOCK_REASONS:
        raise ValueError(f"lock_reason must be one of {sorted(LOCK_REASONS)}")
    old = read_json(config.SPLITS_LOCK)
    if old and reason == "initial_seal":
        raise RuntimeError("splits.lock.json exists; a re-seal needs data_bug_fix or source_refresh")
    if old is None and reason != "initial_seal":
        raise RuntimeError("no lock yet: first seal must be initial_seal")
    d = df.assign(split=assign_split(df["published_date"]))
    lab = d[d["label_12m"].notna()]
    counts = {s: {"n": int((lab["split"] == s).sum()),
                  "n_pos": int(lab.loc[lab["split"] == s, "label_12m"].sum())}
              for s in config.SPLITS}
    lock = {"hash": lock_hash(), "locked_at": now_utc(), "reason": reason, "note": note,
            "values": config.lockable(), "counts": counts, "git": git_state(), **(extra or {})}
    atomic_write_json(config.SPLITS_LOCK, lock)
    append_lock_history({"reason": reason, "note": note, "old_hash": (old or {}).get("hash"),
                         "new_hash": lock["hash"], "git": lock["git"]["commit"]})
    manifest_event("splits_lock", reason=reason, hash=lock["hash"])
    return lock


def require_lock() -> dict:
    lock = read_json(config.SPLITS_LOCK)
    if lock is None:
        raise RuntimeError("splits.lock.json missing: run notebook 02 and seal the splits first")
    if lock["hash"] != lock_hash():
        raise RuntimeError("config.py differs from splits.lock.json; restore it, or delete the lock "
                           "with a documented lock_reason and recompute everything downstream")
    return lock


def temporal_split(df: pd.DataFrame) -> pd.Series:
    require_lock()
    return assign_split(df["published_date"])


def val_halves(df: pd.DataFrame, split: pd.Series) -> pd.Series:
    """val -> val_a (selection, fusion fitting) / val_b (calibration, Conformal)."""
    out = split.copy()
    is_val = split == "val"
    out[is_val & (df["published_date"] <= pd.Timestamp(config.VAL_A_END))] = "val_a"
    out[is_val & (df["published_date"] > pd.Timestamp(config.VAL_A_END))] = "val_b"
    return out


def check_no_time_overlap(df: pd.DataFrame, split: pd.Series) -> None:
    order = ["train", "val_a", "val_b", "val", "test"]
    present = [s for s in order if (split == s).any()]
    for a, b in zip(present, present[1:]):
        if df.loc[split == a, "published_date"].max() >= df.loc[split == b, "published_date"].min():
            raise ValueError(f"time overlap between {a} and {b}")


def random_split(df: pd.DataFrame, seed: int, like: pd.Series) -> pd.Series:
    """Factor A leaky level: same split sizes as `like`, rows shuffled."""
    rng = np.random.default_rng(seed)
    labels = like.dropna().to_numpy().copy()
    rng.shuffle(labels)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[like.dropna().index] = labels
    return out


# ===================================================================== protocol and get_xy
@dataclass(frozen=True)
class Protocol:
    """Four binary leakage factors (section 8). All False = the clean protocol."""
    random_split: bool = False       # A
    current_desc: bool = False       # B
    keep_exploit: bool = False       # C
    keep_zero_days: bool = False     # D

    @property
    def name(self) -> str:
        return "".join(c if v else "-" for c, v in zip("ABCD", asdict(self).values())) or "----"

    @property
    def is_clean(self) -> bool:
        return not any(asdict(self).values())


CLEAN = Protocol()
TEXT_ONLY_NUMERIC = ["desc_len", "n_versions", "n_tokens"]          # derived from text only
FUSED_EXTRA = ["epss_at_pred", "cvss_at_tpred", "cvss_missing_at_tpred", "cwe"]


def assert_no_forbidden(columns) -> None:
    bad = sorted(set(columns) & set(config.FORBIDDEN_FEATURES))
    if bad:
        raise ValueError(f"forbidden features in X: {bad}")


def get_xy(df: pd.DataFrame, protocol: Protocol = CLEAN, feature_set: str = "text_only",
           seed: int = 0, label: str = "label_12m"):
    """Returns X (DataFrame with column 'text' + allowed numeric), y, split.

    Rows: labelled (not censored), split assigned, text not missing. Notebooks never
    build X/y themselves.
    """
    if feature_set not in ("text_only", "fused"):
        raise ValueError("feature_set must be text_only or fused")
    from .features import text_derived
    d = df[df[label].notna()].copy()
    split = val_halves(d, temporal_split(d))
    if not protocol.keep_zero_days:
        d = d[d["is_zero_day"] == 0]
        split = split.loc[d.index]
    if protocol.random_split:
        split = random_split(d, seed, split)
    text = d["desc_current"] if protocol.current_desc else d["desc_original"]
    if not protocol.keep_exploit:
        text = text.map(remove_exploit_sentences)
    keep = split.notna() & text.notna() & (text.str.len() > 0)
    d, split, text = d[keep], split[keep], text[keep]
    X = pd.DataFrame({"text": text}, index=d.index).join(text_derived(text))
    if feature_set == "fused":
        X = X.join(d[FUSED_EXTRA])
    assert_no_forbidden(X.columns)
    y = d[label].astype(int)
    return X, y, split
