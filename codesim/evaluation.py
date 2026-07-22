"""Evaluation utilities: threshold-based metrics and threshold tuning.

The metrics mirror the ones reported in the clone-detection literature so
results from this toolkit line up with published tables: accuracy, precision,
recall, F1, Matthews correlation, and ROC-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    roc_auc: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mcc": self.mcc,
            "roc_auc": self.roc_auc,
            "threshold": self.threshold,
        }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return float(num / den) if den else 0.0


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Rank-sum (Mann-Whitney) form of the area under the ROC curve.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(scores, labels, threshold: float = 0.5) -> Scores:
    """Compute all metrics for continuous ``scores`` at a fixed threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (scores >= threshold).astype(int)
    tp, tn, fp, fn = _confusion(labels, preds)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return Scores(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=_mcc(tp, tn, fp, fn),
        roc_auc=_roc_auc(scores, labels),
        threshold=threshold,
    )


def best_threshold(scores, labels, steps: int = 101) -> Scores:
    """Sweep thresholds in [0, 1] and keep the one with the highest F1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best = None
    for t in np.linspace(0.0, 1.0, steps):
        result = evaluate(scores, labels, float(t))
        if best is None or result.f1 > best.f1:
            best = result
    return best
