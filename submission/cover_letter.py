"""submission/cover_letter.py"""
import re
from . import checks

REQUIRED = ["max_inflation", "max_inflation_lo", "max_inflation_hi",
            "text_pr_auc", "epss_pr_auc"]

def draft() -> str:
    h = checks.load_headline()
    v = h["values"]
    missing = [k for k in REQUIRED if k not in v]
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", h.get("zenodo_doi") or ""):
        missing.append("zenodo_doi")
    if missing:
        raise ValueError(f"Cannot draft cover letter: missing {missing}")
    return f"""Dear Editor,

We submit "What Does the Disclosure Text Really Know? Leakage-Safe Prediction of
Real-World Vulnerability Exploitation from CVE Descriptions" for consideration in
the Journal of Information Security and Applications.

The paper addresses a practical problem for security operations teams: which newly
disclosed vulnerabilities to prioritise. It contributes a leakage-safe evaluation
protocol and quantifies how much common leakage sources inflate reported results.

Key findings:
- The largest single-source inflation factor is {v['max_inflation']:.1f}x
  (95% CI {v['max_inflation_lo']:.1f}-{v['max_inflation_hi']:.1f}).
- Text-only PR-AUC {v['text_pr_auc']:.3f} vs. EPSS-only {v['epss_pr_auc']:.3f}
  on the same test rows.
- We document a leakage path created by NVD's 2026 enrichment policy, where the
  presence of a CVSS score in current records correlates with KEV listing.

Code, data-processing scripts and the dataset are archived on Zenodo ({h['zenodo_doi']}).
The manuscript is original and not under consideration elsewhere.
"""
