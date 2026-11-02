"""Model ladder: TF-IDF (LR, LightGBM) -> frozen embeddings + LightGBM -> fine-tuned
encoders -> QLoRA LLM -> zero-shot LLM. Every fit uses train rows only and every
selection uses val_a only.

TF-IDF/LR is tested offline. GPU parts (encoders, QLoRA, zero-shot) need torch,
transformers, peft and bitsandbytes on Colab and are exercised by the smoke tests in
notebooks 05 and 06; they have not been run in the build environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import fit_tfidf


# ===================================================================== TF-IDF baselines
def tfidf_lr(X: pd.DataFrame, y: pd.Series, split: pd.Series, C: float = 1.0, seed: int = 0,
             class_weight="balanced") -> dict:
    from sklearn.linear_model import LogisticRegression
    vec, mats = fit_tfidf(X, split)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight=class_weight, random_state=seed,
                             solver="liblinear")
    clf.fit(mats["train"], y[split == "train"])
    scores = {s: pd.Series(clf.predict_proba(m)[:, 1], index=X.index[split == s])
              for s, m in mats.items()}
    return {"model": clf, "vectorizer": vec, "scores": scores}


def tfidf_lgbm(X, y, split, seed: int = 0, **params) -> dict:
    import lightgbm as lgb
    vec, mats = fit_tfidf(X, split, max_features=50_000)
    p = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
         "subsample_freq": 1, "colsample_bytree": 0.5, "is_unbalance": True, "seed": seed,
         "deterministic": True, "num_threads": 4, "verbose": -1, **params}
    clf = lgb.LGBMClassifier(**p)
    clf.fit(mats["train"].astype(np.float32), y[split == "train"],
            eval_set=[(mats["val_a"].astype(np.float32), y[split == "val_a"])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    scores = {s: pd.Series(clf.predict_proba(m.astype(np.float32))[:, 1], index=X.index[split == s])
              for s, m in mats.items()}
    return {"model": clf, "vectorizer": vec, "scores": scores}


def tune_on_val_a(fit_fn, grid: list[dict], X, y, split) -> tuple[dict, dict]:
    """Grid search scored on val_a PR-AUC only."""
    from sklearn.metrics import average_precision_score
    best, best_score, all_scores = None, -1.0, {}
    for params in grid:
        res = fit_fn(X, y, split, **params)
        s = average_precision_score(y[split == "val_a"], res["scores"]["val_a"])
        all_scores[str(params)] = s
        if s > best_score:
            best, best_score = params, s
    return best, all_scores


# ===================================================================== embeddings + LightGBM
def embeddings_lgbm(emb: np.ndarray, y: pd.Series, split: pd.Series, seed: int = 0) -> dict:
    import lightgbm as lgb
    idx = {s: np.where((split == s).to_numpy())[0] for s in split.dropna().unique()}
    clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31,
                             colsample_bytree=0.5, is_unbalance=True, seed=seed,
                             deterministic=True, num_threads=4, verbose=-1)
    clf.fit(emb[idx["train"]], y.iloc[idx["train"]],
            eval_set=[(emb[idx["val_a"]], y.iloc[idx["val_a"]])], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    scores = {s: pd.Series(clf.predict_proba(emb[i])[:, 1], index=y.index[i]) for s, i in idx.items()}
    return {"model": clf, "scores": scores}


# ===================================================================== fine-tuned encoders
def finetune_encoder(X, y, split, model_name: str, seed: int = 0, loss: str = "bce",
                     epochs: int = 3, lr: float = 2e-5, batch_size: int = 16,
                     max_length: int = config.MAX_LENGTH, fp16: bool = True,
                     smoke: bool = False, out_dir: str = "/content/ckpt") -> dict:
    """Fine-tune a sequence-classification encoder. loss in {bce, focal, nnpu}.
    smoke=True: 64 training rows, 1 epoch, records peak memory (guide section 10)."""
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments, EarlyStoppingCallback)
    from .losses import focal_loss, nnpu_loss
    from .seeds import seed_everything
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    tr = np.where((split == "train").to_numpy())[0]
    if smoke:
        tr, epochs = tr[:64], 1

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            enc = tok(X["text"].iloc[r], truncation=True, max_length=max_length)
            enc["labels"] = float(y.iloc[r])
            return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, attn_implementation="sdpa") if "ModernBERT" in model_name else \
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    prior = float(y.iloc[tr].mean())

    class LossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            logits = model(**inputs).logits.squeeze(-1)
            if loss == "focal":
                l = focal_loss(logits, labels)
            elif loss == "nnpu":
                l = nnpu_loss(logits, labels, prior=prior)
            else:
                pw = torch.tensor((1 - prior) / max(prior, 1e-6), device=logits.device)
                l = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
            return (l, logits) if return_outputs else l

    from transformers import DataCollatorWithPadding
    va = np.where((split == "val_a").to_numpy())[0]
    args = TrainingArguments(out_dir, num_train_epochs=epochs, learning_rate=lr,
                             per_device_train_batch_size=batch_size,
                             per_device_eval_batch_size=64, fp16=fp16 and torch.cuda.is_available(),
                             eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="eval_loss", seed=seed, data_seed=seed,
                             report_to=[], save_total_limit=1)
    trainer = LossTrainer(model=model, args=args, train_dataset=DS(tr), eval_dataset=DS(va[:2000]),
                          data_collator=DataCollatorWithPadding(tok),
                          callbacks=[EarlyStoppingCallback(1)])
    trainer.train()
    scores = {}
    for s in split.dropna().unique():
        rows = np.where((split == s).to_numpy())[0]
        logits = trainer.predict(DS(rows)).predictions.reshape(-1)
        scores[s] = pd.Series(1 / (1 + np.exp(-logits)), index=y.index[rows])
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    return {"scores": scores, "max_memory_gb": peak, "model_name": model_name}


# ===================================================================== QLoRA
def qlora_classifier(X, y, split, base: str = config.QLORA_BASE, seed: int = 0, epochs: int = 2,
                     lr: float = 2e-4, max_length: int = config.MAX_LENGTH, smoke: bool = False,
                     grad_accum: int = 16, out_dir: str = "/content/qlora") -> dict:
    """4-bit QLoRA, batch 1 + gradient accumulation + gradient checkpointing, fp16 (T4)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig)
    from .seeds import seed_everything
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, quantization_config=bnb, attn_implementation="sdpa")
    model.config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"]))
    tr = np.where((split == "train").to_numpy())[0]
    if smoke:
        tr, epochs = tr[:64], 1
    prior = float(y.iloc[tr].mean())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    pw = torch.tensor((1 - prior) / max(prior, 1e-6), device=model.device)
    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(tr)
        for step, r in enumerate(order, 1):
            enc = tok(X["text"].iloc[r], truncation=True, max_length=max_length,
                      return_tensors="pt").to(model.device)
            logit = model(**enc).logits.squeeze()
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, torch.tensor(float(y.iloc[r]), device=model.device), pos_weight=pw)
            (l / grad_accum).backward()
            if step % grad_accum == 0:
                opt.step(); opt.zero_grad()
    model.eval()
    scores = {}
    with torch.no_grad():
        for s in split.dropna().unique():
            rows = np.where((split == s).to_numpy())[0]
            out = []
            for r in rows:
                enc = tok(X["text"].iloc[r], truncation=True, max_length=max_length,
                          return_tensors="pt").to(model.device)
                out.append(torch.sigmoid(model(**enc).logits.squeeze()).item())
            scores[s] = pd.Series(out, index=y.index[rows])
    return {"scores": scores, "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9}


# ===================================================================== zero-shot LLM
def zero_shot_scores(texts: pd.Series, model_name: str = config.ZERO_SHOT_MODEL,
                     prompt_path=config.ZERO_SHOT_PROMPT, max_length: int = config.MAX_LENGTH) -> pd.Series:
    """Score = P("Yes") / (P("Yes") + P("No")) for the next token. Temperature 0 by
    construction (no sampling). One fixed prompt, hashed into selection.lock.json."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tmpl = prompt_path.read_text(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 device_map="auto").eval()
    yes = tok.encode(" Yes", add_special_tokens=False)[-1]
    no = tok.encode(" No", add_special_tokens=False)[-1]
    out = []
    with torch.no_grad():
        for t in texts:
            short = tok.decode(tok.encode(t, add_special_tokens=False)[:max_length])
            msgs = [{"role": "user", "content": tmpl.format(description=short)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          enable_thinking=False).to(model.device)
            logits = model(ids).logits[0, -1].float()
            pr = torch.softmax(logits[[yes, no]], dim=0)[0].item()
            out.append(pr)
    return pd.Series(out, index=texts.index)
