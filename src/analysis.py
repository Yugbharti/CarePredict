"""Extended analytics -> models/analysis.json, reports/*.csv, reports/figures/*.png"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .data import coerce_features, load_data
from .explain import original_feature, positive_class_shap, transform_df
from .train import make_candidates
from .utils import metrics_at


def boot_auc_ci(y, p, n=1000):
    rng = np.random.default_rng(config.RANDOM_STATE)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=130)
    plt.close(fig)


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METRICS_PATH.read_text())
    sel = meta["selected_model"]

    df = load_data()
    X = coerce_features(df)
    y = df[config.TARGET].values
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    # ------------------------------------------------ 1. out-of-fold predictions, all models
    oof, per_model = {}, {}
    for name, spec in make_candidates().items():
        pp = cross_val_predict(spec["pipe"], X, y, cv=cv, method="predict_proba")[:, 1]
        oof[name] = pp
        per_model[name] = {
            "roc_auc": round(float(roc_auc_score(y, pp)), 3),
            "avg_precision": round(float(average_precision_score(y, pp)), 3),
            "brier": round(float(brier_score_loss(y, pp)), 3),
        }
        print(f"OOF {name:22s} AUC={per_model[name]['roc_auc']}")
    p = oof[sel]

    oof_df = pd.DataFrame({"target": y})
    for name, pp in oof.items():
        oof_df[f"p_{name}"] = np.round(pp, 4)
    oof_df["p_selected"] = np.round(p, 4)
    oof_df.to_csv(config.OOF_CSV, index_label="record_id")

    # ------------------------------------------------ 2. threshold analysis
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    thr_rows = [metrics_at(y, p, t) for t in ts]
    fine = np.round(np.arange(0.01, 0.99, 0.01), 2)
    fine_rows = [metrics_at(y, p, t) for t in fine]

    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    youden = metrics_at(y, p, float(thr[k]))
    ok = [r for r in fine_rows if r["recall"] >= 0.90]
    recall90 = max(ok, key=lambda r: r["threshold"]) if ok else None

    # ------------------------------------------------ 3. figures
    cmp_df = pd.read_csv(config.COMPARISON_PATH).sort_values("roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(cmp_df["model"], cmp_df["roc_auc_mean"], xerr=cmp_df["roc_auc_std"],
            color="#2a9d8f")
    ax.set_xlim(0, 1)
    ax.set_xlabel("CV ROC-AUC (mean +/- SD, 5 folds)")
    ax.set_title("Model comparison (stratified 5-fold CV)")
    save(fig, "model_comparison_cv.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, pp in oof.items():
        f_, t_, _ = roc_curve(y, pp)
        ax.plot(f_, t_, label=f"{name} (AUC {per_model[name]['roc_auc']})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - out-of-fold predictions, all 303 records")
    ax.legend(fontsize=8)
    save(fig, "roc_overlay_oof.png")

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color="#e76f51")
    ax.axhline(y.mean(), color="gray", ls="--", lw=0.8, label="prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall (OOF, {sel})")
    ax.legend()
    save(fig, "pr_curve_oof.png")

    prob_true, prob_pred = calibration_curve(y, p, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, "o-", color="#264653")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction with disease")
    ax.set_title("Calibration (OOF, quantile bins)")
    save(fig, "calibration_oof.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for r_name in ("precision", "recall", "specificity", "f1"):
        ax.plot([r["threshold"] for r in fine_rows], [r[r_name] for r in fine_rows],
                label=r_name)
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_title("Metrics vs threshold (OOF)")
    ax.legend()
    save(fig, "threshold_curves.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p[y == 0], bins=20, alpha=0.65, label="No disease", color="#1f77b4")
    ax.hist(p[y == 1], bins=20, alpha=0.65, label="Disease", color="#d62728")
    ax.set_xlabel("Predicted probability (OOF)")
    ax.set_ylabel("Records")
    ax.set_title("Score separation by true class")
    ax.legend()
    save(fig, "score_distribution.png")

    # ------------------------------------------------ 4. subgroup analysis (threshold 0.5)
    ev = df.copy()
    ev["p"] = p
    groups = {
        "Sex": ev["sex"].map(config.OPTION_LABELS["sex"]),
        "Age band": pd.cut(ev["age"], [0, 49, 59, 200], labels=["<50", "50-59", "60+"]),
        "Chest pain type": ev["cp"].map(config.OPTION_LABELS["cp"]),
    }
    sub_rows = []
    for gname, series in groups.items():
        series = series.astype(str)
        order = ["<50", "50-59", "60+"] if gname == "Age band" else sorted(series.unique())
        for level in order:
            g = ev[series == level]
            if len(g) == 0:
                continue
            yy, pp = g[config.TARGET].values, g["p"].values
            auc = roc_auc_score(yy, pp) if (len(set(yy)) == 2 and len(g) >= 20) else None
            m = metrics_at(yy, pp, 0.5)
            sub_rows.append({
                "attribute": gname, "group": level, "n": int(len(g)),
                "prevalence": round(float(yy.mean()), 3),
                "roc_auc": None if auc is None else round(float(auc), 3),
                "recall": m["recall"], "precision": m["precision"],
                "accuracy": m["accuracy"],
            })

    # ------------------------------------------------ 5. error analysis (threshold 0.5)
    pred = (p >= 0.5).astype(int)
    err = df.copy()
    err["p_disease"] = np.round(p, 3)
    err["error_type"] = np.where((pred == 1) & (y == 0), "False positive",
                         np.where((pred == 0) & (y == 1), "False negative", "Correct"))
    err["confidence_in_error"] = np.round(np.abs(err["p_disease"] - err[config.TARGET]), 3)
    bad = err[err["error_type"] != "Correct"].sort_values("confidence_in_error",
                                                          ascending=False)
    bad.to_csv(config.ERRORS_CSV, index_label="record_id")
    m05 = metrics_at(y, p, 0.5)

    # ------------------------------------------------ 6. SHAP table (grouped by original feature)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    X_t = transform_df(pipe, X)
    sv = positive_class_shap(explainer, X_t.values)
    grouped = pd.DataFrame(0.0, index=X.index, columns=config.FEATURES)
    for j, name in enumerate(X_t.columns):
        grouped[original_feature(name)] += sv[:, j]
    table = pd.DataFrame({"target": y, "p_selected": np.round(p, 4)}, index=X.index)
    for f in config.FEATURES:
        table[f"shap_{f}"] = grouped[f].round(5)
        table[f"val_{f}"] = X[f]
    table.to_csv(config.SHAP_CSV, index_label="record_id")
    ranking = (grouped.abs().mean().sort_values(ascending=False))
    shap_ranking = [{"feature": f, "mean_abs_shap": round(float(v), 4)}
                    for f, v in ranking.items()]

    # ------------------------------------------------ 7. save json
    analysis = {
        "selected_model": sel,
        "n_records": int(len(y)),
        "prevalence": round(float(y.mean()), 3),
        "oof": {
            **per_model[sel],
            "roc_auc_ci95": boot_auc_ci(y, p),
        },
        "per_model_oof": per_model,
        "thresholds": thr_rows,
        "operating_points": {"default_0.5": m05, "youden": youden, "recall_ge_90": recall90},
        "calibration": {"prob_pred": [round(float(v), 3) for v in prob_pred],
                        "prob_true": [round(float(v), 3) for v in prob_true]},
        "subgroups": sub_rows,
        "errors": {"false_negatives": m05["fn"], "false_positives": m05["fp"],
                   "true_positives": m05["tp"], "true_negatives": m05["tn"]},
        "shap_ranking": shap_ranking,
    }
    config.ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2))
    print("Saved", config.ANALYSIS_PATH, config.OOF_CSV, config.SHAP_CSV, config.ERRORS_CSV)


if __name__ == "__main__":
    main()
