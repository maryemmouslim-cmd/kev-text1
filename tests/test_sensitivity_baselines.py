import numpy as np
import pandas as pd
import pytest

from src import baselines, config, labels, sensitivity


def _kev_curve(fracs, n=20):
    rows = []
    for i, f in enumerate(fracs):
        month = pd.Timestamp("2021-11-01") + pd.DateOffset(months=i)
        k = int(round(f * n))
        for j in range(n):
            pub = month - pd.Timedelta(days=800 if j < k else 30)
            rows.append({"cve_id": f"CVE-{i}-{j}", "kev_date_added": month, "published_date": pub})
    d = pd.DataFrame(rows)
    return sensitivity.kev_retroactive_monthly(d[["cve_id", "kev_date_added"]],
                                               d[["cve_id", "published_date"]])


def test_stability_start_first_month_of_lasting_calm():
    m = _kev_curve([.9, .8, .05, .3, .05, .05, .0])
    assert str(sensitivity.stability_start(m)) == "2022-03"


def test_small_months_ignored():
    m = _kev_curve([.9, .05, .05])
    m.loc[m.index[1], "n"] = 2                               # 2 additions: excluded from decision
    assert str(sensitivity.stability_start(m)) == "2022-01"


def test_no_stable_period_falls_back(synth):
    m = _kev_curve([.9, .8, .7])
    with pytest.raises(sensitivity.NoStablePeriod):
        sensitivity.stability_start(m)
    rec = sensitivity.s1_record(m, synth, labels.assign_split(synth["published_date"]))
    assert rec["rule"] == "fallback" and rec["start"] == config.S1_FALLBACK_START
    assert rec["months_dropped"] == 12 and 0 < rec["n_pos_kept"] < rec["n_pos_total"]


def test_embargo_mask(synth):
    split = labels.assign_split(synth["published_date"])
    keep = sensitivity.embargo_mask(synth, split)
    test_start = pd.Timestamp(config.SPLITS["test"][0])
    kept_train = synth[keep & (split == "train")]
    assert (kept_train["published_date"] + pd.Timedelta(days=365) < test_start).all()


def test_summary_table_hides_noncomparable():
    pm = pd.Series({"a": .05, "b": .04, "c": .03})
    res = {"clean": {"per_model": pm}}
    for code in ("S1", "S2"):
        res[code] = {"pr_auc": .045, "ci_lo": .03, "ci_hi": .06, "delta": -.005, "delta_lo": -.01,
                     "delta_hi": .0, "per_model": pm * 0.9, "n_pos": 50}
    t = sensitivity.summary_table(res).set_index("analysis")
    assert t.loc["S1", "pr_auc"] == .045 and np.isnan(t.loc["S2", "pr_auc"])
    assert t.loc["S2", "kendall_tau"] == pytest.approx(1.0)


def test_baselines(synth):
    split = labels.assign_split(synth["published_date"])
    r = baselines.group_rate(synth, split, "vendor")
    assert r[synth["vendor"] == "Acme"].mean() > r[synth["vendor"] == "Hooli"].mean()
    k = baselines.keyword_score(synth["desc_original"])
    assert k[synth["desc_original"].str.contains("remote code", na=False)].min() >= 1
    assert (baselines.cvss_score(synth)[synth["cvss_at_tpred"].isna()] == -1).all()


def test_keyword_rule_has_no_exploit_status_phrase():
    assert not any(labels.has_exploit_phrase(k) for k in config.KEYWORD_RULE)
