import numpy as np
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    roc_auc_score, 
    roc_curve, 
    ndcg_score
)
from scipy.stats import spearmanr


def compute_precision_recall(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    precision, recall, thresholds= precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc


def spearman_correlation(model_scores, human_scores):
    return spearmanr(model_scores, human_scores).correlation


def precision_at_k(y_true, model_scores, k=5):
    """Precision@K based on model ranking"""
    sorted_pairs = sorted(
        zip(model_scores, y_true),
        reverse=True
    )
    top_k = sorted_pairs[:k]
    return sum(label for _, label in top_k) / k


def compute_f1_scores(precision, recall):
    """Calculates F1-score for every point in the PR curve."""
    f1_scores = []
    for p, r in zip(precision, recall):
        if (p + r) == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * (p * r) / (p + r))
    return f1_scores



def compute_f1_vs_threshold(model_scores, y_true, num_thresholds=50):
    thresholds = np.linspace(min(model_scores), max(model_scores), num_thresholds)
    f1_scores = []

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in model_scores]

        tp = sum((p == 1 and h == 1) for p, h in zip(preds, y_true))
        fp = sum((p == 1 and h == 0) for p, h in zip(preds, y_true))
        fn = sum((p == 0 and h == 1) for p, h in zip(preds, y_true))

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        f1_scores.append(f1)

    return thresholds, f1_scores


def compute_roc(model_scores, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, model_scores)
    auc = roc_auc_score(y_true, model_scores)
    return fpr, tpr, auc


def compute_ndcg_at_k(human_scores, model_scores, k=5):
    """
    NDCG@K using graded relevance (human_score 1–5)
    """
    y_true = np.array([human_scores])
    y_score = np.array([model_scores])
    return ndcg_score(y_true, y_score, k=k)