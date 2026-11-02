"""Metrics, paired bootstrap, DeLong, Holm, model selection on val_a, selection lock,
run log and compute budget, headline numbers for the submission layer."""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             matthews_corrcoef, roc_auc_score)

from . import config
from .io_utils import atomic_write_json, atomic_write_text, git_state, now_utc, read_json, sha256_file
from .validate import holm  # noqa: F401  (re-exported)


# ===================================================================== metrics
def precision_at_k(y, p, k: int) -> float:
    idx = np.argsort(-np.asarray(p))[:k]
    return float(np.asarray(y)[idx].mean()) if k else float("nan")


def recall_at_k(y, p, k: int) -> float:
    y = np.asarray(y)
    idx = np.argsort(-np.asarray(p))[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def effort_for_recall(y, p, recall: float = 0.5) -> float:
    """Fraction of CVEs that must be reviewed (in score order) to reach `recall`."""
    y = np.asarray(y)[np.argsort(-np.asarray(p))]
    if y.sum() == 0:
        return float("nan")
    k = int(np.searchsorted(np.cumsum(y), math.ceil(recall * y.sum()))) + 1
    return k / len(y)


def mcc_at(y, p, threshold: float) -> float:
    return float(matthews_corrcoef(y, (np.asarray(p) >= threshold).astype(int)))


def mcc_top_k(y, p, frac: float = config.ALERT_BUDGET) -> float:
    p = np.asarray(p)
    k = max(1, int(round(frac * len(p))))
    pred = np.zeros(len(p), int)
    pred[np.argsort(-p)[:k]] = 1
    return float(matthews_corrcoef(y, pred))


def best_mcc_threshold(y, p) -> float:
    """Chosen on val_a and written to selection.lock.json; never on test."""
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400)))
    return float(max(qs, key=lambda t: mcc_at(y, p, t)))


def ece_equal_mass(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.array_split(np.argsort(p), n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows = [{"p_mean": p[b].mean(), "y_rate": y[b].mean(), "n": len(b)}
            for b in np.array_split(np.argsort(p), n_bins) if len(b)]
    return pd.DataFrame(rows)


def metrics(y, p, threshold: float | None = None) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    base = y.mean()
    k = max(1, int(round(config.ALERT_BUDGET * len(y))))
    out = {
        "n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(base),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "lift": float(average_precision_score(y, p) / base) if base else float("nan"),
        "precision@budget": precision_at_k(y, p, k), "recall@budget": recall_at_k(y, p, k),
        "precision@100": precision_at_k(y, p, 100), "effort@50recall": effort_for_recall(y, p, .5),
        "mcc@budget": mcc_top_k(y, p),
    }
    if threshold is not None:
        out["mcc@threshold"] = mcc_at(y, p, threshold)
    if np.all((p >= 0) & (p <= 1)):
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        out.update(brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, pc, labels=[0, 1])),
                   ece=ece_equal_mass(y, p))
    return out


# ===================================================================== uncertainty
def bootstrap_ci(y, p, metric=average_precision_score, n: int = config.BOOTSTRAP_N,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(metric(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap(y, pa, pb, metric=average_precision_score, n: int = config.BOOTSTRAP_N,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference metric(a) - metric(b) on the same rows and the same resamples."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        diffs.append(metric(y[i], pa[i]) - metric(y[i], pb[i]))
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta": float(metric(y, pa) - metric(y, pb)), "lo": float(lo), "hi": float(hi),
            "p": float(min(p_two, 1.0))}


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, pa, pb) -> dict:
    """DeLong test for two correlated ROC AUCs (fast version, Sun & Xu 2014)."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([pa, pb]).astype(float)
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


# ===================================================================== selection (P5)
def best_by_validation(val_scores: dict[str, float]) -> str:
    """Highest val_a PR-AUC. Test scores are never an argument."""
    return max(val_scores, key=val_scores.get)


def _selection() -> dict:
    return read_json(config.SELECTION_LOCK, default={"models": {}, "best_encoder": None})


def lock_model_selection(model: str, val_a_pr_auc: float, params: dict, mcc_threshold: float,
                         prompt_path=None) -> dict:
    """Freeze one model's hyper-parameters and MCC threshold (both from val_a) before
    any test evaluation of that model. An entry is never overwritten."""
    sel = _selection()
    if model in sel["models"]:
        raise RuntimeError(f"selection for {model!r} exists; delete it only with a documented reason")
    sel["models"][model] = {"locked_at": now_utc(), "val_a_pr_auc": float(val_a_pr_auc),
                            "params": params, "mcc_threshold": float(mcc_threshold),
                            "alert_budget_frac": config.ALERT_BUDGET, "git": git_state()["commit"],
                            "prompt_sha256": sha256_file(prompt_path) if prompt_path else None}
    atomic_write_json(config.SELECTION_LOCK, sel)
    return sel


def lock_best_encoder(candidates: list[str]) -> str:
    """Best fine-tuned encoder by val_a PR-AUC among locked candidates (used in RQ2 and S1-S8)."""
    sel = _selection()
    if sel.get("best_encoder"):
        raise RuntimeError("best_encoder already locked")
    missing = [c for c in candidates if c not in sel["models"]]
    if missing:
        raise RuntimeError(f"lock these models first: {missing}")
    best = best_by_validation({c: sel["models"][c]["val_a_pr_auc"] for c in candidates})
    sel["best_encoder"] = best
    atomic_write_json(config.SELECTION_LOCK, sel)
    return best


def test_metrics(model: str, y, p, *, is_synthetic: bool = False) -> dict:
    """The only function that computes metrics on test rows."""
    entry = _selection()["models"].get(model)
    if entry is None and not is_synthetic:
        raise RuntimeError(f"{model!r} not in selection.lock.json: lock it on val_a before test")
    return metrics(y, p, threshold=(entry or {}).get("mcc_threshold"))


# ===================================================================== run log and budget
@contextmanager
def log_run(model: str, seed: int, protocol: str, is_synthetic: bool, **extra):
    """with log_run(...) as rec: ... ; rec['metrics'] = ...  -> one JSON line per run."""
    rec = {"model": model, "seed": seed, "protocol": protocol, "is_synthetic": is_synthetic,
           "start": now_utc(), "git": git_state(), **extra}
    try:
        import torch
        rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        rec["gpu"] = None
    t = time.time()
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.time() - t, 2)
        try:
            import torch
            if torch.cuda.is_available():
                rec["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except ImportError:
            pass
        log = (config.OUTPUTS_SYNTHETIC if is_synthetic else config.OUTPUTS) / "runs.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def compute_budget(is_synthetic: bool = False) -> pd.DataFrame:
    log = (config.OUTPUTS_SYNTHETIC if is_synthetic else config.OUTPUTS) / "runs.jsonl"
    if not log.exists():
        return pd.DataFrame()
    runs = pd.read_json(log, lines=True)
    runs["gpu_hours"] = np.where(runs["gpu"].notna(), runs["seconds"] / 3600, 0.0)
    return runs.groupby(["model", "protocol"]).agg(runs=("seed", "size"),
                                                  hours=("seconds", lambda s: s.sum() / 3600),
                                                  gpu_hours=("gpu_hours", "sum")).reset_index()


RUN_PLAN = [  # (category, models, n_models, cells, seeds, device)
    ("RQ3 encoders/QLoRA", "modernbert,deberta,securebert,qwen3-qlora", 4, 1, 5, "gpu"),
    ("RQ3 loss variants", "best encoder x {focal, nnpu}", 2, 1, 5, "gpu"),
    ("RQ3 linear", "tfidf-lr,tfidf-lgbm,emb-lgbm", 3, 1, 5, "cpu"),
    ("RQ3 embeddings", "qwen3-embedding encode", 1, 1, 1, "gpu"),
    ("RQ3 zero-shot", "qwen3 zero-shot", 1, 1, 1, "gpu"),
    ("RQ2 factorial", "tfidf-lr", 1, 16, 3, "cpu"),
    ("RQ2 encoder", "best encoder", 1, 6, 3, "gpu"),
    ("Sensitivity", "tfidf-lr + best encoder", 2, len(config.SENSITIVITY_PLAN), 3, "cpu+gpu"),
]


def run_count_table(smoke_minutes: dict | None = None, n_train: int | None = None,
                    epochs: int = 3) -> pd.DataFrame:
    """smoke_minutes: {models string: minutes for 64 samples x 1 epoch} from the smoke tests."""
    rows = []
    for cat, fam, n_models, cells, seeds, dev in RUN_PLAN:
        runs = n_models * cells * seeds
        r = {"category": cat, "models": fam, "runs": runs, "device": dev}
        if smoke_minutes and fam in smoke_minutes and n_train:
            r["est_hours"] = runs * smoke_minutes[fam] * (n_train / 64) * epochs / 60
        rows.append(r)
    return pd.DataFrame(rows)


# ===================================================================== score store
def _score_dir(model: str, is_synthetic: bool):
    return (config.OUTPUTS_SYNTHETIC if is_synthetic else config.OUTPUTS) / "scores" / model


def save_scores(model: str, seed: int, scores: dict, protocol: str = "----",
                is_synthetic: bool = False) -> None:
    """scores: {split: Series indexed by row id}. One file per model x protocol x seed."""
    d = _score_dir(model, is_synthetic)
    d.mkdir(parents=True, exist_ok=True)
    frame = pd.concat({s: v for s, v in scores.items()}, names=["split", "row"]).rename("score")
    frame.to_frame().to_pickle(d / f"{protocol}_seed{seed}.pkl")


def load_scores(model: str, split: str, protocol: str = "----", is_synthetic: bool = False) -> pd.Series:
    """Mean over seeds of one model's scores on one split."""
    files = sorted(_score_dir(model, is_synthetic).glob(f"{protocol}_seed*.pkl"))
    if not files:
        raise FileNotFoundError(f"no scores for {model} {protocol}")
    per_seed = [pd.read_pickle(f)["score"].xs(split, level="split") for f in files]
    return pd.concat(per_seed, axis=1).mean(axis=1)


def list_models(is_synthetic: bool = False) -> list[str]:
    root = (config.OUTPUTS_SYNTHETIC if is_synthetic else config.OUTPUTS) / "scores"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def lock_from_val_a(model: str, y: pd.Series, params: dict, protocol: str = "----",
                    is_synthetic: bool = False, prompt_path=None) -> dict:
    """Seed-averaged val_a scores -> PR-AUC + MCC threshold -> selection.lock.json."""
    va = load_scores(model, "val_a", protocol, is_synthetic)
    yv = y.loc[va.index]
    return lock_model_selection(model, average_precision_score(yv, va), params,
                                best_mcc_threshold(yv.to_numpy(), va.to_numpy()), prompt_path)


# ===================================================================== headline numbers
def _headline_path():
    return config.OUTPUTS / "headline_numbers.json"


def write_headline_numbers(entries: dict, is_synthetic: bool) -> None:
    """entries: {"max_inflation": (4.27, "outputs/tables/rq2_inflation.csv", "factor=B")}"""
    if is_synthetic:
        raise RuntimeError("Synthetic results never go to outputs/headline_numbers.json")
    g = git_state()
    if g["dirty"]:
        raise RuntimeError("Commit code changes (src/, notebooks/, submission/, tests/) "
                           "before writing headline numbers")
    for name, (v, table, _) in entries.items():
        if v is None or math.isnan(v):
            raise ValueError(f"{name} is missing")
        if not (config.ROOT / table).exists():
            raise FileNotFoundError(f"{name}: source table {table} not found")
    path = _headline_path()
    h = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
         else {"values": {}, "provenance": {}, "zenodo_doi": None})
    for name, (v, table, loc) in entries.items():
        h["values"][name] = float(v)
        h["provenance"][name] = {"table": table, "locator": loc, "git_hash": g["commit"]}
    h["is_synthetic"] = False
    atomic_write_text(path, json.dumps(h, indent=2))


def set_zenodo_doi(doi: str) -> None:
    path = _headline_path()
    h = json.loads(path.read_text(encoding="utf-8"))
    h["zenodo_doi"] = doi
    atomic_write_text(path, json.dumps(h, indent=2))


def save_table(df: pd.DataFrame, name: str, is_synthetic: bool) -> str:
    """Final tables: refuse synthetic rows in outputs/, and dirty code."""
    if not is_synthetic:
        if "is_synthetic" in df and df["is_synthetic"].any():
            raise RuntimeError("synthetic rows in a final table")
        if git_state()["dirty"]:
            raise RuntimeError("commit code before saving a final table")
    root = config.OUTPUTS_SYNTHETIC if is_synthetic else config.OUTPUTS
    path = root / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return str(path.relative_to(config.ROOT))
