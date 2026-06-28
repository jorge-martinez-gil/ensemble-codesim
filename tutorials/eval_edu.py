# -*- coding: utf-8 -*-
"""
eval_edu — minimal, dependency-light evaluation helpers for the tutorials.

Everything here is pure NumPy + standard library so the notebooks run without
scikit-learn. If you *do* have scikit-learn installed, its versions of these
metrics will agree with ours; we re-implement them only to keep the classroom
setup friction-free.

Contents
--------
  roc_auc(scores, labels)                 rank-based ROC-AUC
  metrics_at(scores, labels, threshold)   accuracy / precision / recall / F1 / MCC
  best_threshold_f1(scores, labels)       sweep thresholds, return best F1 + details
  mean_ensemble(X)                        average the (normalised) measures
  vote_ensemble(X, thresholds)            majority vote of per-measure decisions
  logreg_fit_predict(...)                 a learned linear ensemble (pure NumPy)
  load_feature_file(path)                 read the repo's outputs/*.txt feature rows

The order of the 21 production measures (as written by exec-bigclonebench.py /
exec-karnalim.py) is recorded in MEASURE_NAMES_21.
"""

from __future__ import annotations

import json

import numpy as np

# Column order of the 21 measures in outputs/output-*.txt (last column = label).
MEASURE_NAMES_21 = [
    "ast", "bow", "codebert", "comments", "exe", "fcall", "fuzz", "graph",
    "hashing", "image", "jaccard", "lcs", "lev", "metrics", "ngrams", "pdg",
    "rk", "semclone", "semdiff", "tdf", "winn",
]


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

def roc_auc(scores, labels):
    """Rank-based ROC-AUC (a.k.a. the probability a random positive outranks a
    random negative). Ties are handled with average ranks.

    Returns ``float('nan')`` if labels are all positive or all negative.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _counts(pred, labels):
    pred = np.asarray(pred, dtype=int)
    labels = np.asarray(labels, dtype=int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return tp, tn, fp, fn


def metrics_at(scores, labels, threshold):
    """Compute accuracy/precision/recall/F1/MCC at a decision ``threshold``."""
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp, tn, fp, fn = _counts(pred, labels)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    denom = math_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def math_sqrt(x):
    """sqrt that returns 0.0 for non-positive input (avoids import noise)."""
    return float(np.sqrt(x)) if x > 0 else 0.0


def best_threshold_f1(scores, labels):
    """Return the metrics dict whose F1 is highest over all decision thresholds.

    Implemented as a single O(n log n) sweep: sort by descending score and, for
    every prefix "predict the top-k pairs as clones", read off tp/fp from
    cumulative sums. Tied scores are kept in the same group so a threshold never
    splits equal scores. This stays fast even on the 25k-row BigCloneBench file.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(scores)
    P = int((labels == 1).sum())
    if n == 0:
        return metrics_at(scores, labels, 0.0)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    lab = labels[order]
    cum_tp = np.cumsum(lab == 1)
    cum_fp = np.cumsum(lab == 0)
    # End-of-tie-group positions: keep only the last index of each equal-score run.
    boundary = np.ones(n, dtype=bool)
    boundary[:-1] = s[1:] != s[:-1]
    idx = np.where(boundary)[0]
    tp = cum_tp[idx].astype(float)
    k = (idx + 1).astype(float)
    precision = np.where(k > 0, tp / k, 0.0)
    recall = tp / P if P > 0 else np.zeros_like(tp)
    denom = precision + recall
    f1 = np.zeros_like(precision)
    nz = denom > 0
    f1[nz] = 2 * precision[nz] * recall[nz] / denom[nz]
    best_i = int(np.argmax(f1))
    best_thr = s[idx[best_i]]
    return metrics_at(scores, labels, best_thr)


# --------------------------------------------------------------------------- #
# Simple, transparent ensembles
# --------------------------------------------------------------------------- #

def _minmax_columns(X):
    """Min-max scale each column of X to [0, 1] (so measures are comparable)."""
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span


def mean_ensemble(X):
    """Average of the min-max-normalised measures -> one score per pair."""
    return _minmax_columns(X).mean(axis=1)


def vote_ensemble(X, thresholds):
    """Fraction of measures that fire (score >= their own threshold).

    ``thresholds`` is a length-n_features array, typically each measure's own
    best-F1 threshold learned on a training split.
    """
    X = np.asarray(X, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    fires = (X >= thresholds).astype(int)
    return fires.mean(axis=1)


# --------------------------------------------------------------------------- #
# A learned linear ensemble (pure NumPy logistic regression)
# --------------------------------------------------------------------------- #

def train_test_split_idx(n, test_frac=0.3, seed=0):
    """Return (train_idx, test_idx) using a reproducible shuffle."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * (1 - test_frac)))
    return perm[:cut], perm[cut:]


def logreg_fit_predict(X_train, y_train, X_test, iters=500, lr=0.5, l2=1e-3):
    """Train a logistic-regression ensemble of the measures with gradient
    descent (pure NumPy) and return predicted probabilities for X_test.

    Features are z-scored using the TRAIN statistics so the test set never
    informs scaling. This is the simplest possible *learned* combination of the
    measures and is enough to show that learning weights beats any single one.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    n, d = Xtr.shape
    Xtr1 = np.hstack([np.ones((n, 1)), Xtr])
    Xte1 = np.hstack([np.ones((len(Xte), 1)), Xte])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xtr1 @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xtr1.T @ (p - y_train) / n
        grad[1:] += l2 * w[1:]          # L2 on weights, not bias
        w -= lr * grad
    z = Xte1 @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_feature_file(path, max_rows=None):
    """Load a repo feature file (one JSON list per line: 21 measures + label).

    Returns ``(X, y, names)`` where X is (n, 21), y is (n,), and names is
    MEASURE_NAMES_21. Rows that do not have exactly 22 numbers are skipped.
    Pass ``max_rows`` to read only the first N valid rows (handy for the large
    BigCloneBench file).
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(vals) == 22:
                rows.append(vals)
                if max_rows and len(rows) >= max_rows:
                    break
    arr = np.array(rows, dtype=float)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)
    return X, y, MEASURE_NAMES_21


if __name__ == "__main__":
    # tiny self-check
    s = [0.1, 0.4, 0.35, 0.8]
    y = [0, 0, 1, 1]
    print("AUC:", round(roc_auc(s, y), 3))
    print("best F1:", round(best_threshold_f1(s, y)["f1"], 3))
