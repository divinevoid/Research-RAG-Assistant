import json
import os
import matplotlib.pyplot as plt

from eval.metrics import (
    compute_precision_recall,
    compute_f1_scores,
    compute_f1_vs_threshold,
    compute_roc,
    compute_ndcg_at_k,
    spearman_correlation,
    precision_at_k
)
from eval.plots import (
    plot_precision_recall,
    plot_f1_vs_threshold,
    plot_roc_curve
)


def load_logs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(log_path):
    logs = load_logs(log_path)

    threshold = logs[0].get("threshold_used", "unknown")

    y_true = [row["human_binary"] for row in logs]
    model_scores = [row["model_score"] for row in logs]
    human_scores = [row["human_score"] for row in logs]

    precision, recall, pr_auc = compute_precision_recall(
        y_true, model_scores
    )

    f1_scores = compute_f1_scores(precision, recall)

    thresholds, f1_vs_t = compute_f1_vs_threshold(model_scores, y_true, num_thresholds=50)
    fpr, tpr, roc_auc = compute_roc(model_scores, y_true)
    spearman = spearman_correlation(model_scores, human_scores)
    p_at_5 = precision_at_k(y_true, model_scores, k=5)

    # ---- SAVE METRICS ----
    os.makedirs("eval_outputs", exist_ok=True)

    metrics_out = {
        "threshold": threshold,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Spearman": spearman,
        "Precision@5": p_at_5,
        "NDCG@5": compute_ndcg_at_k(human_scores, model_scores, k=5),
        "Max_F1": max(f1_scores)
    }

    os.makedirs("eval_outputs", exist_ok=True)

    with open(f"eval_outputs/metrics_threshold_{threshold}.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    plot_precision_recall(precision, recall).savefig(
        f"eval_outputs/pr_curve+{threshold}.png"
    )
    # plot_f1_vs_threshold(threshold, f1_vs_t, float(threshold)).savefig(
    #     f"eval_outputs/f1_vs_threshold_{threshold}.png"
    # )
    fig_f1 = plot_f1_vs_threshold(
            thresholds,          # <-- array of thresholds
            f1_vs_t,             # <-- array of F1 values
            float(threshold)     # <-- chosen operating threshold
        )

    fig_f1.savefig(
    f"eval_outputs/f1_vs_threshold_{threshold}.png",
    dpi=300,
    bbox_inches="tight"
)
    plt.close(fig_f1)
    plot_roc_curve(fpr, tpr, roc_auc).savefig(
        f"eval_outputs/roc_curve_{threshold}.png"
    )

    plt.close("all")

    print(f"✅ Evaluation completed for threshold {threshold}")
    print(metrics_out)

    # metrics_path = f"eval_outputs/metrics_threshold_{threshold}.json"
    # with open(metrics_path, "w") as f:
    #     json.dump(metrics_out, f, indent=2)

    # # ---- SAVE PLOT ----
    # fig = plot_precision_recall(precision, recall)
    # plot_path = f"eval_outputs/pr_curve_threshold_{threshold}.png"
    # fig.savefig(plot_path)
    # plt.close(fig)

    # print(f"✅ Evaluation completed for threshold {threshold}")
    # print(metrics_out)


if __name__ == "__main__":
    run_evaluation("evaluation_logs_0.3.json")
    run_evaluation("evaluation_logs_0.4.json")
    run_evaluation("evaluation_logs_0.5.json")
