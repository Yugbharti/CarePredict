"""Small shared metric helpers (used by analysis.py and app.py)."""
import numpy as np
from sklearn.metrics import confusion_matrix


def metrics_at(y, p, t):
    """Confusion-matrix metrics for probability array p at threshold t."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": round(float(t), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "specificity": round(float(spec), 3),
        "f1": round(float(f1), 3),
        "accuracy": round(float((tp + tn) / len(y)), 3),
    }
