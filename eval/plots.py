import matplotlib.pyplot as plt

def plot_precision_recall(precision, recall):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recall, precision, marker='.', label='PR Curve')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig


def plot_f1_vs_threshold(thresholds, f1_scores, chosen_threshold):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, f1_scores, linewidth=2)
    ax.axvline(
        chosen_threshold,
        linestyle='--',
        linewidth=2,
        label=f"Chosen Threshold = {chosen_threshold}"
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_roc_curve(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig