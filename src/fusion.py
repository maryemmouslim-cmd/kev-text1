"""RQ4: text + EPSS (+ CVSS, CWE) fusion, calibration, Brier, Mondrian Conformal.

Protocol (guide sections 5 and 9):
  * comparison rows = test rows with an EPSS score at t_pred and lag <= EPSS_MAX_LAG_DAYS;
  * fusion model fitted on val_a out-of-sample text scores;
  * Platt / isotonic and Conformal fitted on val_b;
  * test is evaluated once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def comparison_rows(df: pd.DataFrame, idx: pd.Index) -> tuple[pd.Index, dict]:
    d = df.loc[idx]
    ok = d["epss_at_pred"].notna() & (d["epss_lag_days"] <= config.EPSS_MAX_LAG_DAYS)
    return idx[ok.to_numpy()], {"n": len(idx), "no_epss": int(d["epss_at_pred"].isna().sum()),
                                "lag_exceeded": int((d["epss_lag_days"] > config.EPSS_MAX_LAG_DAYS).sum())}


def _design(text_score: pd.Series, df: pd.DataFrame, use_cvss_cwe: bool, cwe_levels=None):
    d = df.loc[text_score.index]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = pd.DataFrame({"text_logit": logit(text_score.to_numpy()),
                      "epss_logit": logit(d["epss_at_pred"].to_numpy())}, index=d.index)
    if use_cvss_cwe:
        X["cvss"] = d["cvss_at_tpred"].fillna(0) / 10
        X["cvss_missing"] = d["cvss_missing_at_tpred"].astype(float)
        if cwe_levels is not None:
            for c in cwe_levels:
                X[f"cwe={c}"] = (d["cwe"] == c).astype(float)
    return X


class Fusion:
    """Logistic stacker. variant: 'epss', 'text', 'text+epss', 'text+epss+cvss+cwe'."""

    def __init__(self, variant: str = "text+epss", seed: int = 0, top_cwe: int = 10):
        self.variant, self.seed, self.top_cwe = variant, seed, top_cwe

    def _cols(self, X):
        if self.variant == "epss":
            return ["epss_logit"]
        if self.variant == "text":
            return ["text_logit"]
        return list(X.columns)

    def fit(self, text_val_a: pd.Series, y_val_a: pd.Series, df: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression
        full = self.variant.endswith("cwe")
        self.cwe_levels = (df.loc[text_val_a.index, "cwe"].value_counts().index[:self.top_cwe]
                           if full else None)
        X = _design(text_val_a, df, full, self.cwe_levels)
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)
        self.clf.fit(X[self._cols(X)], y_val_a)
        return self

    def predict(self, text_score: pd.Series, df: pd.DataFrame) -> pd.Series:
        X = _design(text_score, df, self.variant.endswith("cwe"), self.cwe_levels)
        return pd.Series(self.clf.predict_proba(X[self._cols(X)])[:, 1], index=X.index)


# ---------------------------------------------------------------- calibration
class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method

    def fit(self, p, y):
        p, y = np.asarray(p), np.asarray(y)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        else:   # platt on the logit
            from sklearn.linear_model import LogisticRegression
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
        return self

    def transform(self, p):
        p = np.asarray(p)
        if self.method == "isotonic":
            return self.m.predict(p)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.m.predict_proba(z.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------- Mondrian Conformal
class MondrianConformal:
    """Class-conditional split conformal for binary labels. Nonconformity = 1 - p(label).
    alpha per class from config.conformal_alpha(n_pos_val_b) (locked)."""

    def fit(self, p_cal, y_cal, alpha: dict[int, float]):
        p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal).astype(int)
        self.alpha, self.q = alpha, {}
        for c in (0, 1):
            s = 1 - (p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c])
            n = len(s)
            k = int(np.ceil((n + 1) * (1 - alpha[c])))
            self.q[c] = np.inf if k > n else float(np.sort(s)[k - 1])   # vacuous if too few
        return self

    def predict_sets(self, p) -> np.ndarray:
        """Returns an (n, 2) boolean array: [contains 0, contains 1]."""
        p = np.asarray(p)
        # nonconformity of label 0 is 1 - (1 - p) = p; of label 1 it is 1 - p
        return np.column_stack([p <= self.q[0], (1 - p) <= self.q[1]])

    def coverage(self, p, y) -> dict:
        sets, y = self.predict_sets(p), np.asarray(y).astype(int)
        cov = {f"coverage_{c}": float(sets[y == c, c].mean()) if (y == c).any() else np.nan
               for c in (0, 1)}
        cov["mean_set_size"] = float(sets.sum(1).mean())
        cov["empty_sets"] = float((sets.sum(1) == 0).mean())
        return cov
