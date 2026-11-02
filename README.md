# kev-text

**What Does the Disclosure Text Really Know? Leakage-Safe Prediction of Real-World
Vulnerability Exploitation from CVE Descriptions**

Predicts whether a CVE is added to CISA KEV within 12 months of publication, from the
description available at disclosure (t_pred = t0 + 1 day), and measures how much four
leakage factors inflate results (2x2x2x2 design).

## Layout

```text
src/            science code (never imports submission/)
  config.py       every constant; LOCKED values are frozen by splits.lock.json
  io_utils.py     resumable downloads, manifest.json + hashes, atomic writes, git state
  data.py         cvelistV5, NVD change history, NVD CVE API, KEV (+ Wayback), EPSS; merge
  cvss.py         CVSS v3.x base score from a vector
  labels.py       label_12m, censoring, zero-days, exploit-sentence removal, split lock, get_xy
  validate.py     Table 2, restoration bias (Fisher/chi2 + Holm), temporal profile,
                  vendor/CWE reports, near duplicates (MinHash), length report
  features.py     TF-IDF fitted on train only, DAPT guard, entity masking, embeddings
  models.py       TF-IDF LR/LightGBM, embeddings+LightGBM, encoders, QLoRA, zero-shot
  losses.py       focal, nnPU
  leakage.py      16 cells, encoder cells, inflation ratios, factorial effects
  fusion.py       RQ4 stacker, isotonic/Platt, Mondrian Conformal
  evaluate.py     metrics (PR-AUC, lift, MCC@budget, ECE), bootstrap, DeLong, selection
                  lock, run log, compute budget, score store, headline numbers
  sensitivity.py  S1-S8, KEV retroactive curve
  baselines.py    CVSS, CWE/vendor rates (train only), keyword rule
  explain.py      exact linear SHAP, Integrated Gradients
  plots.py        300 dpi figures; synthetic -> outputs_synthetic/ + watermark
  seeds.py        seed_everything
  synthetic.py    synthetic data for tests and dry runs only
submission/     JISA packaging (reads outputs/headline_numbers.json only)
notebooks/      01-11, thin: they call src/ and never build X/y themselves
tests/          pytest suite
prompts/        zero-shot prompt (hashed into selection.lock.json)
```

## Run order

| Step | Notebook | Device | Writes |
| --- | --- | --- | --- |
| 1 | 01_collect (SAMPLE = True first, then full) | CPU | data/raw, data/dataset.parquet, manifest.json |
| 2 | 02_clean_eda_lock | CPU | splits.lock.json, dataset_labelled.parquet, Figure 8 |
| 3 | 03_baselines | CPU | tfidf-lr + non-learning baselines in selection.lock.json |
| 4 | 08_leakage_rq2 (run 1, TF-IDF) | CPU | RQ2 tables |
| 5 | 04_embeddings, 05_encoders (SMOKE = True first) | GPU | encoder scores, best_encoder |
| 6 | 08_leakage_rq2 (run 2, best encoder) | GPU | RQ2 encoder tables |
| 7 | 06_llm (SMOKE = True first) | GPU | QLoRA, zero-shot |
| 8 | 07_rq3_test | CPU | Table RQ3 (test, once) |
| 9 | 09_fusion_rq4 | CPU | Table RQ4, calibration, Conformal |
| 10 | 10_explain, 11_sensitivity | CPU/GPU | SHAP, Table 5.5 |

Set `SYN = True` in any notebook from 03 on for a dry run on synthetic data
(outputs go to outputs_synthetic/).

## Guards that stop the pipeline

- `labels.require_lock()`: no splits.lock.json, or config.py changed after the lock.
- `labels.write_lock()`: re-seal only with `data_bug_fix` or `source_refresh`; every seal is
  appended to lock_history.jsonl.
- `labels.assert_no_forbidden()`: any column in `config.FORBIDDEN_FEATURES` inside X.
- `features.fit_tfidf()`: vocabulary and IDF from train rows only.
- `features.assert_dapt_corpus()`: no text published after TRAIN_END in a pretraining corpus.
- `evaluate.test_metrics()`: refuses a model that is not in selection.lock.json (val_a).
- `evaluate.save_table()`, `plots.save_figure()`, `evaluate.write_headline_numbers()`:
  refuse synthetic rows and uncommitted code for final outputs.
- `validate.assert_ready()`: duplicate cve_id, raw files changed after download, or
  desc_original restored for under 50 % of rows.

## Tests

```bash
pip install -r requirements.txt -r submission/requirements.txt pytest
pytest -v
```

99 tests cover labels and censoring edge cases, the split lock, TF-IDF fitting, parsers
(on fixtures shaped like each API), CVSS scores, metrics, DeLong, Conformal coverage, the
factorial, sensitivity rules, the submission layer, and an end-to-end synthetic run.

## Not yet verified against real sources (check on day 1)

- NVD change history: `data.parse_history_events` assumes `CVSS V3.1` / `CWE` details with
  the source as the first word of `newValue` (e.g. `NIST AV:N/...`). `cvss_at_tpred`
  depends on it; check 10 CVEs before the full pull.
- KEV history: cisagov/kev-data starts on 2025-01-27. `data.kev_wayback_snapshots` lists
  Wayback captures; if 2022-2024 is not covered, drop S6 and say so in Threats to Validity.
- EPSS host: `config.EPSS_DAILY_URLS` tries two hosts in order.
- GPU code (encoders, QLoRA, zero-shot, embeddings) has not been executed in the build
  environment; the SMOKE runs in notebooks 04-06 are its first execution.

## Before submission

```python
from submission import checks, journal
journal.mark_done("guide_for_authors_read", "read on <date>; word limit = ...")
checks.check_draft(HIGHLIGHTS, ABSTRACT)     # while writing
checks.assert_ready(HIGHLIGHTS, ABSTRACT)    # only right before Submit
```
