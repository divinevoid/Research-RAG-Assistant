import numpy as np

'''def find_optimal_threshold(model_scores, human_scores):
    thresholds = np.linspace(0.1, 0.9, 30)
    best_threshold = 0
    best_score = 0

    for t in thresholds:
        selected = [
            h for s, h in zip(model_scores, human_scores)
            if s >= t
        ]

        if len(selected) == 0:
            continue

        avg_score = np.mean(selected)

        if avg_score > best_score:
            best_score = avg_score
            best_threshold = t

    return best_threshold, best_score'''

def find_optimal_threshold(model_scores, human_binary):
    thresholds = np.linspace(min(model_scores), max(model_scores), 50)
    best_threshold = thresholds[0]
    best_f1 = -1

    for t in thresholds:
        # Predict 1 if score >= t, else 0
        preds = [1 if s >= t else 0 for s in model_scores]
        
        # Calculate Precision and Recall manually
        tp = sum((p == 1 and h == 1) for p, h in zip(preds, human_binary))
        fp = sum((p == 1 and h == 0) for p, h in zip(preds, human_binary))
        fn = sum((p == 0 and h == 1) for p, h in zip(preds, human_binary))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1