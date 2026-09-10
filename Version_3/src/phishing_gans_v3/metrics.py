from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def select_threshold(labels: np.ndarray, scores: np.ndarray, maximum_fpr: float = 0.01) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    positives = max(1, int((labels == 1).sum()))
    negatives = max(1, int((labels == 0).sum()))
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    fpr = fp / negatives
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    candidates = np.flatnonzero(distinct & (fpr <= maximum_fpr))
    index = int(candidates[np.argmax(f1[candidates])]) if len(candidates) else int(np.argmin(fpr))
    return {
        "threshold": float(sorted_scores[index]), "validation_f1": float(f1[index]),
        "validation_precision": float(precision[index]), "validation_recall": float(recall[index]),
        "validation_fpr": float(fpr[index]), "maximum_fpr_constraint": maximum_fpr,
    }


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
