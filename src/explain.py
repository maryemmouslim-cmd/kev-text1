"""Explanations. For TF-IDF + LR the SHAP values are exact and closed-form under feature
independence: phi_j = w_j * (x_j - E[x_j]). Integrated Gradients for fine-tuned encoders
uses captum (Colab). Explanations describe the model, not causes of exploitation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def linear_shap(model, vectorizer, X_train_mat, X_mat) -> np.ndarray:
    mean = np.asarray(X_train_mat.mean(axis=0)).ravel()
    w = model.coef_.ravel()
    return (X_mat.toarray() - mean) * w if hasattr(X_mat, "toarray") else (X_mat - mean) * w


def top_terms(model, vectorizer, X_train_mat, X_pos_mat, k: int = 30) -> pd.DataFrame:
    """Mean SHAP over positive test rows (by n-gram) - the RQ4 pattern table."""
    names = np.array(vectorizer.get_feature_names_out())
    mean = np.asarray(X_train_mat.mean(axis=0)).ravel()
    w = model.coef_.ravel()
    contrib = (np.asarray(X_pos_mat.mean(axis=0)).ravel() - mean) * w
    order = np.argsort(-contrib)
    return pd.DataFrame({"term": names[order[:k]], "mean_shap": contrib[order[:k]],
                         "weight": w[order[:k]]})


def integrated_gradients(model, tokenizer, text: str, max_length: int = 256, steps: int = 50):
    from captum.attr import LayerIntegratedGradients
    import torch
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    emb = model.get_input_embeddings()
    fwd = lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
    lig = LayerIntegratedGradients(fwd, emb)
    base = torch.full_like(enc["input_ids"], tokenizer.pad_token_id or 0)
    attr = lig.attribute(enc["input_ids"], baselines=base,
                         additional_forward_args=(enc["attention_mask"],), n_steps=steps)
    scores = attr.sum(-1).squeeze(0).detach().cpu().numpy()
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return pd.DataFrame({"token": toks, "attribution": scores})
