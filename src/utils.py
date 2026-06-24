"""Shared helpers: seeding, metrics, threshold tuning."""
from __future__ import annotations
import random
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, confusion_matrix)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "ap": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    return out


def best_threshold(y_true, y_prob) -> float:
    """Threshold maximizing F1 over a sweep."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)
