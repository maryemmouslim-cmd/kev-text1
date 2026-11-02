"""Figures at 300 dpi. Synthetic data -> outputs_synthetic/ with a SYNTHETIC watermark;
final figures refuse synthetic rows and uncommitted code."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .io_utils import git_state  # noqa: E402


def save_figure(fig, name: str, data: pd.DataFrame | None, is_synthetic: bool) -> str:
    synthetic_rows = data is not None and "is_synthetic" in data and bool(data["is_synthetic"].any())
    if synthetic_rows and not is_synthetic:
        raise RuntimeError("synthetic rows in a final figure")
    if is_synthetic or synthetic_rows:
        fig.text(0.5, 0.5, "SYNTHETIC", fontsize=48, color="red", alpha=0.25,
                 ha="center", va="center", rotation=30, transform=fig.transFigure)
        root = config.OUTPUTS_SYNTHETIC
    else:
        if git_state()["dirty"]:
            raise RuntimeError("commit code before saving a final figure")
        root = config.OUTPUTS
    path = root / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def kev_retroactive_curve(m: pd.DataFrame, start=None):
    """Figure 8: monthly KEV additions (bars) and retroactive fraction (line)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = m.index.to_timestamp()
    ax.bar(x, m["n"], width=20, color="#c9ced6", label="KEV additions per month")
    ax.set_ylabel("Additions")
    ax2 = ax.twinx()
    ax2.plot(x, m["frac"], color="#1f4e79", marker="o", ms=3, label="Retroactive fraction")
    ax2.axhline(config.STABILITY_THRESHOLD, color="#b03a2e", ls="--", lw=1, label="Threshold")
    if start is not None:
        ax2.axvline(pd.Timestamp(start), color="#b03a2e", lw=1)
    ax2.set_ylabel("Published > 365 days before addition")
    ax2.set_ylim(0, 1)
    fig.legend(loc="upper right", fontsize=8, frameon=False)
    return fig


def reliability_diagram(tables: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lim = max(max(t["p_mean"].max(), t["y_rate"].max()) for t in tables.values()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=0.8, ls=":")
    for name, t in tables.items():
        ax.plot(t["p_mean"], t["y_rate"], marker="o", ms=3, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed KEV rate")
    ax.legend(fontsize=8, frameon=False)
    return fig


def inflation_bars(effects: pd.DataFrame):
    """Main-effect inflation factors from leakage.factorial_effects."""
    e = effects[~effects["term"].str.contains(":") & (effects["term"] != "intercept")]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(e["term"], e["inflation_factor"], color="#1f4e79")
    ax.axvline(1, color="grey", lw=0.8)
    ax.set_xlabel("Inflation factor (x)")
    return fig


def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], base_rate: float):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (rec, prec) in curves.items():
        ax.plot(rec, prec, label=name, lw=1.2)
    ax.axhline(base_rate, color="grey", ls=":", lw=0.8, label="Random")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=8, frameon=False)
    return fig
