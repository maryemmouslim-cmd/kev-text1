"""Focal loss and non-negative PU loss (Kiryo et al., NeurIPS 2017) on logits."""
from __future__ import annotations


def focal_loss(logits, labels, gamma: float = 2.0, alpha: float = 0.25):
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * labels + (1 - p) * (1 - labels)
    a_t = alpha * labels + (1 - alpha) * (1 - labels)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def nnpu_loss(logits, labels, prior: float, beta: float = 0.0, gamma: float = 1.0):
    """labels: 1 = KEV positive, 0 = unlabelled (not 'negative': KEV is incomplete).
    prior = class prior of positives among all CVEs; the observed KEV rate is a lower
    bound, so S2-style checks report sensitivity to it.
    Sigmoid surrogate loss l(z, y) = sigmoid(-y z)."""
    import torch
    pos, unl = labels == 1, labels == 0
    n_p, n_u = pos.sum().clamp(min=1), unl.sum().clamp(min=1)
    l_pos = torch.sigmoid(-logits)          # loss of predicting +1
    l_neg = torch.sigmoid(logits)           # loss of predicting -1
    r_p_plus = (l_pos * pos).sum() / n_p
    r_p_minus = (l_neg * pos).sum() / n_p
    r_u_minus = (l_neg * unl).sum() / n_u
    neg_risk = r_u_minus - prior * r_p_minus
    if neg_risk < -beta:
        # Kiryo et al., Algorithm 1: when the negative risk goes below -beta, step on
        # -gamma * neg_risk instead of the full risk (pushes it back towards zero).
        return -gamma * neg_risk
    return prior * r_p_plus + neg_risk
