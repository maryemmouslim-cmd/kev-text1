"""Central configuration. Every constant the paper depends on lives here.

Values marked LOCKED are written into splits.lock.json by notebook 02 and may not
change afterwards without deleting the lock (see labels.write_lock / lock_reason).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
MANIFEST = DATA / "manifest.json"
DATASET = DATA / "dataset.parquet"
OUTPUTS = ROOT / "outputs"
OUTPUTS_SYNTHETIC = ROOT / "outputs_synthetic"
SPLITS_LOCK = ROOT / "splits.lock.json"
SELECTION_LOCK = ROOT / "selection.lock.json"
LOCK_HISTORY = ROOT / "lock_history.jsonl"
RUN_LOG = OUTPUTS / "runs.jsonl"
PROMPTS = ROOT / "prompts"

# ---------------------------------------------------------------- sources
NVD_CVE_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_HISTORY_API = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"
NVD_MAX_WINDOW_DAYS = 120            # NVD limit for changeStartDate/changeEndDate
NVD_PAGE_SIZE = 5000
NVD_API_KEY_ENV = "NVD_API_KEY"      # optional; raises the rate limit
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
KEV_WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
# EPSS daily files. The host moved in the past; the first that answers is used and logged.
EPSS_DAILY_URLS = (
    "https://epss.empiricalsecurity.com/epss_scores-{date}.csv.gz",
    "https://epss.cyentia.com/epss_scores-{date}.csv.gz",
)
CVELIST_REPO = "https://github.com/CVEProject/cvelistV5.git"

# ---------------------------------------------------------------- time definitions
KEV_LAUNCH = pd.Timestamp("2021-11-03")
T_PRED_OFFSET_DAYS = 1               # t_pred = t0 + 1 day
LABEL_HORIZON_DAYS = 365             # label_12m
ZERO_DAY_DAYS = 7                    # is_zero_day: KEV add - t0 <= 7 days (negative included)
EPSS_MAX_LAG_DAYS = 7                # rows beyond this lag are excluded from RQ4 and counted
assert ZERO_DAY_DAYS >= T_PRED_OFFSET_DAYS, "labels observable at t_pred would leak"

# LOCKED: initial split boundaries (inclusive). Fixed in notebook 02 after seeing counts.
SPLITS = {
    "train": ("2022-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2024-06-30"),
    "test":  ("2024-07-01", "2025-06-30"),
}
VAL_A_END = "2024-03-31"             # LOCKED: val_a = val[..VAL_A_END], val_b = the rest
TRAIN_START = pd.Timestamp(SPLITS["train"][0])
TRAIN_END = pd.Timestamp(SPLITS["train"][1])

# Label embargo (S8): train labels observable before the test period starts.
EMBARGO_DAYS = LABEL_HORIZON_DAYS

# ---------------------------------------------------------------- leakage protocol
FORBIDDEN_FEATURES = [
    "desc_current", "is_zero_day", "has_exploit_phrase", "kev_date_added",
    "label_12m", "label_any_kev", "followup_complete", "followup_days",
    "cvss_present_now", "nvd_status_now", "cvss_current", "cwe_current",
    "epss_current", "kev_vendor_project", "kev_product", "kev_short_description",
    "kev_notes", "kev_ransomware", "adp_content",
]
# Sentences containing any of these are removed when factor C is at its clean level.
EXPLOIT_PHRASES = [
    r"exploited in the wild", r"actively exploited", r"active exploitation",
    r"exploitation (?:has been|was) (?:observed|detected|reported)",
    r"(?:is|are|has been|have been) (?:being )?exploited",
    r"known to be exploited", r"under active attack", r"in-the-wild",
    r"limited,? targeted attacks?", r"aware of (?:reports|a report) that an exploit",
    r"zero-day", r"0-day",
]
# Non-learning keyword baseline (P6). LOCKED. No exploitation-status phrases here.
KEYWORD_RULE = [
    "remote code execution", "execute arbitrary code", "arbitrary code execution",
    "authentication bypass", "bypass authentication", "unauthenticated",
    "command injection", "privilege escalation", "elevation of privilege",
    "deserialization", "path traversal", "sql injection",
]

# ---------------------------------------------------------------- sensitivity plan (LOCKED)
STABILITY_THRESHOLD = 0.10
STABILITY_MIN_N = 5
S1_FALLBACK_START = "2023-01-01"
SENSITIVITY_PLAN = {
    "S1": "train only from the KEV stability start",
    "S2": "label_any_kev (ranking only)",
    "S3": "horizon 6 months (and 24 months where follow-up allows; ranking only)",
    "S4": "t_pred = t0 + 7 days",
    "S5": "selection effect of missing desc_original, measured with EPSS (all vs restored rows)",
    "S6": "labels from union of historical KEV snapshots (ranking only)",
    "S7": "entity masking: vendor, product and version names",
    "S8": "label embargo: train labels observable before test start",
}
COMPARABLE = {"S1", "S4", "S7", "S8"}   # same labels and test rows as clean (S5 reported separately)

# ---------------------------------------------------------------- seeds and budget
SEEDS_FINAL = [0, 1, 2, 3, 4]
SEEDS_FACTORIAL = [0, 1, 2]
SEED_EXPLORE = 0
BOOTSTRAP_N = 10_000
ALERT_BUDGET = 0.01                  # top 1 % for MCC@budget and precision@budget

# ---------------------------------------------------------------- models
MAX_LENGTH = 256                     # LOCKED after validate.length_report
ENCODERS = {
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta": "microsoft/deberta-v3-base",
    "securebert": "ehsanaghaei/SecureBERT",
}
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QLORA_BASE = "Qwen/Qwen3-1.7B"
QLORA_STACK = "peft"                 # "peft" or "unsloth"; decided by smoke test
ZERO_SHOT_MODEL = "Qwen/Qwen3-1.7B"
ZERO_SHOT_PROMPT = PROMPTS / "zero_shot_v1.txt"

# ---------------------------------------------------------------- Conformal (LOCKED via n_pos)
def conformal_alpha(n_pos_val_b: int) -> dict[int, float]:
    return {0: 0.10, 1: 0.10 if n_pos_val_b >= 40 else 0.20}


def lockable() -> dict:
    """Everything that splits.lock.json freezes. Changing any value changes the hash."""
    return {
        "SPLITS": SPLITS, "VAL_A_END": VAL_A_END,
        "T_PRED_OFFSET_DAYS": T_PRED_OFFSET_DAYS, "LABEL_HORIZON_DAYS": LABEL_HORIZON_DAYS,
        "ZERO_DAY_DAYS": ZERO_DAY_DAYS, "EPSS_MAX_LAG_DAYS": EPSS_MAX_LAG_DAYS,
        "MAX_LENGTH": MAX_LENGTH, "STABILITY_THRESHOLD": STABILITY_THRESHOLD,
        "STABILITY_MIN_N": STABILITY_MIN_N, "S1_FALLBACK_START": S1_FALLBACK_START,
        "SENSITIVITY_PLAN": SENSITIVITY_PLAN, "KEYWORD_RULE": KEYWORD_RULE,
        "EXPLOIT_PHRASES": EXPLOIT_PHRASES,
    }
