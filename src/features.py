"""Text preprocessing, TF-IDF fitted on train only, text-derived features, entity masking,
and frozen embeddings."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

FORBIDDEN_FEATURES = config.FORBIDDEN_FEATURES

_VERSION_TOKEN = re.compile(r"\b(?:v|version\s*)?\d+(?:\.\d+){1,3}[a-z0-9-]*\b", re.I)
_WORD = re.compile(r"\S+")


# ---------------------------------------------------------------- preprocessing
def preprocess(text: str) -> str:
    """Light normalisation only. Negations and words like 'unpatched' are kept."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)                  # stray HTML
    text = re.sub(r"https?://\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_derived(text: pd.Series) -> pd.DataFrame:
    """Features computed from desc_original only (RQ1/RQ3 may use these)."""
    t = text.fillna("")
    return pd.DataFrame({
        "desc_len": t.str.len(),
        "n_versions": t.map(lambda s: len(_VERSION_TOKEN.findall(s))),
        "n_tokens": t.map(lambda s: len(_WORD.findall(s))),
    }, index=text.index)


# ---------------------------------------------------------------- TF-IDF (P3)
def fit_tfidf(X: pd.DataFrame, split: pd.Series, text_col: str = "text", fit_on: str = "train",
              **kw):
    """Vocabulary and IDF learned from split == fit_on only; every split is transformed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    mask = split == fit_on
    if not mask.any():
        raise ValueError(f"no rows with split == {fit_on!r}")
    params = {"ngram_range": (1, 2), "min_df": 2, "max_features": 200_000,
              "sublinear_tf": True, "preprocessor": preprocess, **kw}
    vec = TfidfVectorizer(**params).fit(X.loc[mask, text_col])
    mats = {s: vec.transform(X.loc[split == s, text_col]) for s in split.dropna().unique()}
    return vec, mats


def assert_dapt_corpus(corpus: pd.DataFrame) -> None:
    """Domain-adaptive pretraining corpus: nothing published after the end of training."""
    late = corpus["published_date"] > config.TRAIN_END
    if late.any():
        raise ValueError(f"{int(late.sum())} DAPT texts published after TRAIN_END")


# ---------------------------------------------------------------- entity masking (S7)
def mask_entities(text: str, vendors: list[str], products: list[str]) -> str:
    if not isinstance(text, str):
        return text
    vendors = [v for v in (vendors or []) if isinstance(v, str)]
    products = [p for p in (products or []) if isinstance(p, str)]
    for name in sorted(set(vendors + products), key=len, reverse=True):     # longest first
        if len(name) >= 3:
            tag = "VENDOR" if name in vendors else "PRODUCT"
            text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", tag, text, flags=re.I)
    return _VERSION_TOKEN.sub("VERSION", text)


def mask_frame(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    src = df.loc[X.index]
    out["text"] = [mask_entities(t, v, p) for t, v, p in
                   zip(out["text"], src["vendors"], src["products"])]
    return out


# ---------------------------------------------------------------- frozen embeddings
def embed(texts: pd.Series, model_name: str = config.EMBED_MODEL, batch_size: int = 32,
          max_length: int = config.MAX_LENGTH) -> np.ndarray:
    """Pretrained encoder, no fitting on our data: encoding test texts is not snooping.
    Anything fitted on top (scaling, PCA, LightGBM) must use train only."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    m.max_seq_length = max_length
    return m.encode(texts.map(preprocess).tolist(), batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=True)


def token_lengths(texts: pd.Series, tokenizer=None) -> pd.Series:
    if tokenizer is None:
        return texts.fillna("").map(lambda s: len(_WORD.findall(s)))
    return texts.fillna("").map(lambda s: len(tokenizer(s, add_special_tokens=True)["input_ids"]))
