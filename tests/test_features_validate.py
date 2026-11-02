import numpy as np
import pandas as pd
import pytest

from src import config, features, validate


def test_tfidf_fits_on_train_only():
    X = pd.DataFrame({"text": ["alpha beta gamma", "alpha beta delta", "zeta eta theta", "zeta eta iota"]})
    split = pd.Series(["train", "train", "test", "test"])
    vec, mats = features.fit_tfidf(X, split, min_df=1, ngram_range=(1, 1))
    assert "alpha" in vec.vocabulary_ and "zeta" not in vec.vocabulary_
    X2 = X.copy(); X2.loc[2:, "text"] = ["other words here", "more other words"]
    vec2, _ = features.fit_tfidf(X2, split, min_df=1, ngram_range=(1, 1))
    assert np.allclose(vec.idf_, vec2.idf_)               # test rows do not move IDF


def test_tfidf_needs_train_rows():
    with pytest.raises(ValueError):
        features.fit_tfidf(pd.DataFrame({"text": ["a"]}), pd.Series(["test"]))


def test_dapt_corpus_guard():
    c = pd.DataFrame({"published_date": [config.TRAIN_END, config.TRAIN_END + pd.Timedelta(days=1)]})
    with pytest.raises(ValueError, match="DAPT"):
        features.assert_dapt_corpus(c)
    features.assert_dapt_corpus(c.iloc[:1])


def test_mask_entities():
    t = "Apache Log4j 2.14.1 allows remote attackers to execute code. Fixed in version 2.15."
    m = features.mask_entities(t, ["Apache Software Foundation", "Apache"], ["Log4j"])
    assert m == "VENDOR PRODUCT VERSION allows remote attackers to execute code. Fixed in VERSION."


def test_mask_entities_word_boundaries_and_short_names():
    m = features.mask_entities("HP printers and HPE servers", ["HP"], ["HPE servers"])
    assert m == "HP printers and PRODUCT"                 # names < 3 chars are not masked


def test_text_derived_empty_and_long():
    f = features.text_derived(pd.Series(["", "Fixed in 1.2.3 and 4.5", None]))
    assert f["desc_len"].tolist() == [0, 22, 0] and f["n_versions"].tolist() == [0, 2, 0]


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03], index=list("abc"))
    assert validate.holm(p).round(4).to_dict() == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_wilson_bounds():
    lo, hi = validate.wilson(np.array([0, 5]), np.array([10, 10]))
    assert lo[0] == 0 and 0 < hi[0] < 0.35 and lo[1] < 0.5 < hi[1]


def test_restoration_bias_detects_skew():
    n = 400
    df = pd.DataFrame({"published_date": pd.Timestamp("2022-06-01"),
                       "label_12m": [1] * 40 + [0] * (n - 40),
                       "desc_original": [None] * 20 + ["x"] * 20 + [None] * 10 + ["x"] * (n - 50)})
    r = validate.restoration_bias(df)
    assert r.loc["all", "p"] < 0.001 and r.loc["all", "miss_rate_pos"] == 0.5


def test_near_duplicates():
    base = "A buffer overflow in the image parser of Foo Viewer allows remote attackers to run code"
    texts = pd.Series([base, "Completely different text about a cross site scripting issue in Bar",
                       base + ".", "SQL injection in the login form of Baz CMS via the user field"])
    split = pd.Series(["train", "train", "test", "test"])
    r = validate.near_duplicate_report(texts, split)
    assert r["share_with_near_duplicate"] == 0.5


def test_group_report_train_only(synth):
    r = validate.group_report(synth, "vendor", min_n=30)
    assert {"lift", "lift_lo", "lift_hi"} <= set(r.columns)
    assert r.loc["Acme", "lift"] > r.loc["Hooli", "lift"]


def test_qc_report_and_gate(sandbox, synth):
    rep = validate.qc_report(synth)
    assert rep["duplicate_cve_id"] == 0 and "temporal_profile" in rep
    validate.assert_ready(rep)
    bad = pd.concat([synth, synth.iloc[:1]])
    with pytest.raises(RuntimeError, match="duplicate"):
        validate.assert_ready(validate.qc_report(bad))
