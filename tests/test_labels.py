"""Edge cases from guide section 10: dates, censoring, zero-days, split lock, forbidden columns."""
import pandas as pd
import pytest

from src import config, labels

SNAP = pd.Timestamp("2026-10-16")


def _row(t0, kev=None, **kw):
    return {"cve_id": kw.pop("cve_id", f"CVE-X-{t0}-{kev}"), "published_date": pd.Timestamp(t0),
            "kev_date_added": pd.Timestamp(kev) if kev else pd.NaT, **kw}


def _lab(*rows, snap=SNAP):
    return labels.add_labels(pd.DataFrame(rows), snap)


def test_exactly_365_is_positive_366_negative():
    d = _lab(_row("2023-01-10", "2024-01-10"), _row("2023-01-10", "2024-01-11"))
    assert list(d["label_12m"]) == [1, 0]


def test_feb_29():
    d = _lab(_row("2024-02-29", "2025-02-28"), _row("2024-02-29", "2025-03-01"))
    assert list(d["label_12m"]) == [1, 0]


def test_added_before_publication_is_positive_and_zero_day():
    d = _lab(_row("2023-05-10", "2023-05-01"))
    assert d["label_12m"].iloc[0] == 1 and d["is_zero_day"].iloc[0] == 1
    assert bool(d["kev_before_publication"].iloc[0])


def test_zero_day_boundary():
    d = _lab(_row("2023-05-10", "2023-05-17"), _row("2023-05-10", "2023-05-18"))
    assert list(d["is_zero_day"]) == [1, 0]


def test_window_ending_on_snapshot_day_is_censored():
    snap = pd.Timestamp("2025-01-10")
    d = _lab(_row("2024-01-11"), _row("2024-01-10"), _row("2024-01-09"), snap=snap)
    # 2024-01-11 + 365 = 2025-01-10 = snapshot day -> censored; earlier ones complete
    assert d["label_12m"].isna().tolist() == [True, False, False]


def test_censored_positive_is_excluded_too():
    d = _lab(_row("2026-06-01", "2026-07-01"))
    assert pd.isna(d["label_12m"].iloc[0])


def test_after_kev_launch():
    with pytest.raises(ValueError, match="before KEV"):
        labels.assert_after_kev_launch(pd.DataFrame([_row("2020-05-01")]))


def test_split_boundary_inclusive():
    s = labels.assign_split(pd.Series(pd.to_datetime(["2023-12-31", "2024-01-01", "2025-06-30", "2025-07-01"])))
    assert s.tolist()[:3] == ["train", "val", "test"] and pd.isna(s.iloc[3])


def test_temporal_split_requires_lock(sandbox):
    df = pd.DataFrame([_row("2023-01-01")])
    with pytest.raises(RuntimeError, match="missing"):
        labels.temporal_split(df)


def test_lock_detects_config_change(sandbox, monkeypatch):
    df = _lab(_row("2023-01-01", "2023-03-01"), _row("2024-08-01"))
    labels.write_lock(df, "initial_seal", "first seal")
    labels.temporal_split(df)
    monkeypatch.setattr(config, "SPLITS", {**config.SPLITS, "train": ("2022-01-01", "2023-11-30")})
    with pytest.raises(RuntimeError, match="differs"):
        labels.temporal_split(df)


def test_lock_reasons(sandbox):
    df = _lab(_row("2023-01-01"))
    with pytest.raises(ValueError):
        labels.write_lock(df, "because", "x")
    labels.write_lock(df, "initial_seal", "x")
    with pytest.raises(RuntimeError, match="re-seal"):
        labels.write_lock(df, "initial_seal", "again")
    labels.write_lock(df, "data_bug_fix", "fixed parser")
    assert len(config.LOCK_HISTORY.read_text().splitlines()) == 2


def test_forbidden_rejected():
    for col in config.FORBIDDEN_FEATURES:
        with pytest.raises(ValueError):
            labels.assert_no_forbidden(["text", col])


def test_remove_exploit_sentence_in_the_middle():
    t = "A flaw allows RCE. Microsoft is aware of limited, targeted attacks. It affects 1.2."
    assert labels.remove_exploit_sentences(t) == "A flaw allows RCE. It affects 1.2."
    assert labels.has_exploit_phrase(t) == 1


def test_remove_exploit_handles_empty_and_unicode():
    assert labels.remove_exploit_sentences("") == ""
    assert labels.remove_exploit_sentences(None) is None
    t = "Überprüfung fehlt in Zürich-Modul. Exploited in the wild."
    assert labels.remove_exploit_sentences(t) == "Überprüfung fehlt in Zürich-Modul."


def test_get_xy_clean_protocol(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "test")
    X, y, split = labels.get_xy(synth, labels.CLEAN)
    d = synth.loc[X.index]
    assert d["is_zero_day"].sum() == 0
    assert not X["text"].str.contains("targeted attacks").any()
    assert set(split.unique()) <= {"train", "val_a", "val_b", "test"}
    labels.check_no_time_overlap(d, split)
    assert "desc_current" not in X and "epss_at_pred" not in X


def test_get_xy_removes_exploit_sentences_outside_zero_days(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "test")
    d = synth.copy()
    i = d.index[(d["is_zero_day"] == 0) & d["label_12m"].notna() & d["desc_original"].notna()][:5]
    d.loc[i, "desc_original"] = d.loc[i, "desc_original"] + " Exploitation has been observed in the wild."
    X, _, _ = labels.get_xy(d, labels.CLEAN)
    assert not X["text"].str.contains("observed in the wild").any()
    Xc, _, _ = labels.get_xy(d, labels.Protocol(keep_exploit=True))
    assert Xc["text"].str.contains("observed in the wild").sum() == len(set(i) & set(Xc.index))


def test_get_xy_leaky_cells(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "test")
    Xc, yc, _ = labels.get_xy(synth, labels.CLEAN)
    Xd, yd, _ = labels.get_xy(synth, labels.Protocol(keep_zero_days=True))
    assert yd.sum() > yc.sum()
    Xb, _, _ = labels.get_xy(synth, labels.Protocol(current_desc=True, keep_exploit=True))
    assert Xb["text"].str.contains("actively exploited").any()
    Xa, _, sa = labels.get_xy(synth, labels.Protocol(random_split=True), seed=1)
    with pytest.raises(ValueError, match="overlap"):
        labels.check_no_time_overlap(synth.loc[Xa.index], sa)


def test_get_xy_fused_has_epss(sandbox, synth):
    labels.write_lock(synth, "initial_seal", "test")
    X, _, _ = labels.get_xy(synth, feature_set="fused")
    assert {"epss_at_pred", "cvss_at_tpred", "cwe"} <= set(X.columns)
