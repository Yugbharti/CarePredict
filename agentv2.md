# agent_v2_analytics_upgrade.md — CarePredict: Analytics Dashboard + Full Results Report

> **Prerequisite:** the original `agent.md` is fully completed (Phases 1–10 pass, `python -m pytest -q` → 3 passed).
> **This file upgrades the project.** Do the steps in order. Copy code exactly. Run all commands from the repo root.

## 0. What this upgrade does

| Area | Before (agent.md) | After (this file) |
|---|---|---|
| Frontend | 4 simple tabs, big warning banner, "Limitations & bias" tab | 7-tab **analytics dashboard**: KPI cards, interactive EDA, model lab, threshold explorer, SHAP dependence plots, what-if simulator, subgroup/error analysis, full report viewer |
| Limitations | Tab + warning banner in the app | **Removed from the frontend.** Only a one-line footer remains (see note below) |
| Evaluation | CV table + 61-row hold-out test | + out-of-fold (OOF) predictions on all 303 records, PR curve, calibration, Brier score, bootstrap CIs, threshold sweep, operating points, subgroup performance, error analysis |
| Report | none | Auto-generated **`reports/REPORT.md`** (full results report with tables and figures), viewable and downloadable in the app |

> **Note on "remove limitations":** the dashboard no longer shows a limitations tab or warning banner. A short, factual "Notes on interpretation" section is kept **only inside `REPORT.md` / `MODEL_CARD.md`**, because the project's resume line says limits and the not-a-diagnosis statement are documented. If the owner explicitly wants it gone from the report too, delete the section titled `## 10. Notes on interpretation` in `src/report.py` (and update the resume line accordingly). Do not delete it on your own.

### Rules
1. Do not rename anything. Other files import by name.
2. After each step, run its check. Do not continue if a check fails (see Section 12).
3. Never print or invent numbers in prose by hand — the report generator reads them from files.

### New run order (replaces the old one)
```bash
python -m src.eda
python -m src.train
python -m src.explain
python -m src.analysis      # NEW
python -m src.report        # NEW
streamlit run app.py
```

---

## STEP U1 — Append new paths to `src/config.py`

Add these lines **at the end of the "paths" block** (anywhere after `MODEL_CARD_PATH`):

```python
ANALYSIS_PATH = MODEL_DIR / "analysis.json"
OOF_CSV = REPORT_DIR / "oof_predictions.csv"
SHAP_CSV = REPORT_DIR / "shap_table.csv"
ERRORS_CSV = REPORT_DIR / "misclassified.csv"
REPORT_MD = REPORT_DIR / "REPORT.md"
```

**Check:** `python -c "from src import config; print(config.REPORT_MD)"` prints a path.

---

## STEP U2 — Create `src/utils.py`

```python
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
```

**Check:** `python -c "from src.utils import metrics_at; print(metrics_at([0,1,1,0],[0.2,0.9,0.4,0.6],0.5))"` prints a dict with `tp: 1, fp: 1, fn: 1, tn: 1`.

---

## STEP U3 — Create `src/analysis.py` (extended analytics)

What it does: out-of-fold predictions for **all four models** on all 303 records (each record is predicted by a model that never saw it), bootstrap CIs, PR/calibration/threshold analysis, operating points, subgroup and error analysis, and a SHAP table for interactive plots.

```python
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
```

**Run:** `python -m src.analysis`

**Check (all must be true):**
- `models/analysis.json`, `reports/oof_predictions.csv`, `reports/shap_table.csv`, `reports/misclassified.csv` exist.
- `reports/figures/` now also has: `model_comparison_cv.png`, `roc_overlay_oof.png`, `pr_curve_oof.png`, `calibration_oof.png`, `threshold_curves.png`, `score_distribution.png`.
- Printed OOF AUC for `dummy_most_frequent` ≈ 0.5, for real models ≈ 0.85–0.92.

---

## STEP U4 — Create `src/report.py` (full results report generator)

```python
"""Generate reports/REPORT.md - a full results report built only from saved artefacts."""
import json

import pandas as pd

from . import config
from .data import load_data


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("n/a" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


def fig(name: str, caption: str) -> str:
    return f"![{caption}](figures/{name})\n" if (config.FIG_DIR / name).exists() else ""


def main():
    m = json.loads(config.METRICS_PATH.read_text())
    a = json.loads(config.ANALYSIS_PATH.read_text())
    cmp_ = pd.read_csv(config.COMPARISON_PATH)
    df = load_data()
    T = config.TARGET
    n, prev = len(df), df[T].mean()
    sel, tm, oof = m["selected_model"], m["test_metrics"], a["oof"]
    op = a["operating_points"]

    sel_row = cmp_[cmp_.model == sel].iloc[0]
    lr_row = cmp_[cmp_.model == "logistic_regression"].iloc[0]
    dummy_row = cmp_[cmp_.model == "dummy_most_frequent"].iloc[0]
    diff = sel_row.roc_auc_mean - lr_row.roc_auc_mean
    within = abs(diff) < max(sel_row.roc_auc_std, lr_row.roc_auc_std)
    top3 = ", ".join(config.FEATURE_LABELS[r["feature"]] for r in a["shap_ranking"][:3])

    L = []
    L.append("# CarePredict - Full Results Report\n")
    L.append(f"*Generated {m['trained_at_utc']} UTC - model `{sel}` - "
             "public de-identified UCI Heart Disease (Cleveland) data.*\n")

    # 1 ---------------------------------------------------------------
    L.append("## 1. Executive summary\n")
    L.append(f"- **Task:** supervised binary classification of angiographic heart-disease "
             f"presence from {len(config.FEATURES)} clinical features on {n} records "
             f"({prev:.1%} positive).")
    L.append(f"- **Selected model:** `{sel}` (best tree-based model by cross-validated ROC-AUC).")
    L.append(f"- **Cross-validated ROC-AUC:** {sel_row.roc_auc_mean:.3f} +/- "
             f"{sel_row.roc_auc_std:.3f} (stratified 5-fold, training split).")
    L.append(f"- **Hold-out test ROC-AUC:** {tm['roc_auc']} "
             f"(95% bootstrap CI {tm['roc_auc_ci95'][0]}-{tm['roc_auc_ci95'][1]}, n={m['n_test']}).")
    L.append(f"- **Out-of-fold ROC-AUC on all {n} records:** {oof['roc_auc']} "
             f"(95% CI {oof['roc_auc_ci95'][0]}-{oof['roc_auc_ci95'][1]}); "
             f"average precision {oof['avg_precision']}; Brier score {oof['brier']}.")
    L.append(f"- **Versus baselines:** a majority-class dummy scores {dummy_row.roc_auc_mean:.3f} AUC; "
             f"logistic regression scores {lr_row.roc_auc_mean:.3f}. The selected model differs from "
             f"logistic regression by {diff:+.3f} AUC, which is "
             f"{'within' if within else 'outside'} one fold-to-fold standard deviation.")
    L.append(f"- **Most influential features (mean |SHAP|):** {top3}.")
    L.append(f"- **Operating point (threshold 0.5):** recall {op['default_0.5']['recall']}, "
             f"precision {op['default_0.5']['precision']}, specificity {op['default_0.5']['specificity']}.\n")

    # 2 ---------------------------------------------------------------
    L.append("## 2. Data and preprocessing\n")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    L.append(f"- Source: UCI Machine Learning Repository, Heart Disease, Cleveland subset ({n} rows, "
             f"{len(config.FEATURES)} features). Target = `num > 0`.")
    L.append(f"- Sex: {(df.sex == 1).mean():.1%} male. Age: mean {df.age.mean():.1f}, "
             f"range {int(df.age.min())}-{int(df.age.max())}.")
    L.append("- Missing values: " + ", ".join(f"`{c}` ({int(v)})" for c, v in miss.items())
             + ". Numeric -> median imputation; categorical -> most-frequent imputation.")
    L.append("- Encoding: one-hot (`handle_unknown='ignore'`) for 8 categorical/ordinal-coded features.")
    L.append("- Scaling: `StandardScaler` for logistic regression only; tree models are scale-invariant.")
    L.append("- All preprocessing sits inside the sklearn `Pipeline`, so cross-validation has no leakage.\n")

    # 3 ---------------------------------------------------------------
    L.append("## 3. Exploratory data analysis\n")
    corr = (df[config.NUMERIC + [T]].corr(method="spearman")[T].drop(T)
            .sort_values(key=lambda s: s.abs(), ascending=False).round(3))
    L.append("**Spearman correlation of numeric features with the outcome**\n")
    L.append(md_table(pd.DataFrame({"feature": corr.index, "spearman_rho": corr.values})) + "\n")
    rows = []
    for c in config.CATEGORICAL:
        g = df.groupby(c)[T].agg(["mean", "size"])
        g = g[g["size"] >= 10]
        if len(g) < 2:
            continue
        hi, lo = g["mean"].idxmax(), g["mean"].idxmin()
        rows.append({
            "feature": c,
            "highest-rate level": f"{config.display_value(c, hi)} ({g.loc[hi, 'mean']:.0%}, n={int(g.loc[hi, 'size'])})",
            "lowest-rate level": f"{config.display_value(c, lo)} ({g.loc[lo, 'mean']:.0%}, n={int(g.loc[lo, 'size'])})",
            "spread": round(float(g.loc[hi, "mean"] - g.loc[lo, "mean"]), 2),
        })
    L.append("**Disease rate by categorical level (levels with n >= 10)**\n")
    L.append(md_table(pd.DataFrame(rows).sort_values("spread", ascending=False)) + "\n")
    for f_, c_ in [("eda_target_balance.png", "Target balance"),
                   ("eda_numeric_by_target.png", "Numeric features by outcome"),
                   ("eda_categorical_rates.png", "Disease rate by categorical level"),
                   ("eda_correlation.png", "Correlation heatmap")]:
        L.append(fig(f_, c_))

    # 4 ---------------------------------------------------------------
    L.append("## 4. Methodology\n")
    L.append(f"- Split: stratified {int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)} "
             f"hold-out (train n={m['n_train']}, test n={m['n_test']}), seed {config.RANDOM_STATE}.")
    L.append(f"- Model selection: stratified {config.CV_FOLDS}-fold CV on the training split only; "
             "test set evaluated once.")
    L.append("- Candidates: majority-class dummy; logistic regression (balanced class weights); "
             "random forest (300 trees, min leaf 3); gradient boosting (150 trees, depth 2, lr 0.05).")
    L.append("- Out-of-fold (OOF) analysis: every record is predicted by a model that never saw it, "
             "giving a more stable estimate than the small test split.")
    L.append("- Interpretability: permutation importance (test set) and SHAP TreeExplainer "
             "(contributions of one-hot columns summed back to the original feature).\n")

    # 5 ---------------------------------------------------------------
    L.append("## 5. Model comparison\n")
    show = cmp_[["model", "roc_auc_mean", "roc_auc_std", "recall_mean", "precision_mean",
                 "f1_mean", "accuracy_mean"]].copy()
    show["OOF AUC (all 303)"] = [a["per_model_oof"][x]["roc_auc"] for x in show["model"]]
    L.append(md_table(show) + "\n")
    L.append(fig("model_comparison_cv.png", "Cross-validated ROC-AUC by model"))
    L.append(fig("roc_overlay_oof.png", "OOF ROC curves, all models"))

    # 6 ---------------------------------------------------------------
    L.append("## 6. Final model performance\n")
    L.append("| Evaluation | ROC-AUC | Recall | Precision | F1 | Accuracy |\n|---|---|---|---|---|---|")
    L.append(f"| Hold-out test (n={m['n_test']}) | {tm['roc_auc']} | {tm['recall']} | "
             f"{tm['precision']} | {tm['f1']} | {tm['accuracy']} |")
    d5 = op["default_0.5"]
    L.append(f"| OOF, all {n} records (t=0.5) | {oof['roc_auc']} | {d5['recall']} | "
             f"{d5['precision']} | {d5['f1']} | {d5['accuracy']} |\n")
    L.append(f"Brier score (OOF): {oof['brier']}. Average precision (OOF): {oof['avg_precision']}.\n")
    for f_, c_ in [("roc_curve.png", "ROC curve, hold-out test"),
                   ("confusion_matrix.png", "Confusion matrix, hold-out test"),
                   ("pr_curve_oof.png", "Precision-recall curve (OOF)"),
                   ("calibration_oof.png", "Calibration curve (OOF)"),
                   ("score_distribution.png", "Score distribution by true class")]:
        L.append(fig(f_, c_))

    # 7 ---------------------------------------------------------------
    L.append("## 7. Threshold analysis and operating points\n")
    keep = {0.3, 0.4, 0.5, 0.6, 0.7}
    tt = pd.DataFrame([r for r in a["thresholds"] if round(r["threshold"], 2) in keep])
    L.append(md_table(tt[["threshold", "tp", "fp", "fn", "tn", "precision", "recall",
                          "specificity", "f1"]]) + "\n")
    ops = [("Default (0.5)", op["default_0.5"]), ("Youden's J (max sensitivity+specificity)", op["youden"])]
    if op["recall_ge_90"]:
        ops.append(("Highest threshold with recall >= 0.90", op["recall_ge_90"]))
    L.append(md_table(pd.DataFrame([{"operating point": k, **v} for k, v in ops])) + "\n")
    L.append(fig("threshold_curves.png", "Metrics versus decision threshold"))

    # 8 ---------------------------------------------------------------
    L.append("## 8. Subgroup performance (OOF, threshold 0.5)\n")
    L.append(md_table(pd.DataFrame(a["subgroups"])) + "\n")
    L.append("AUC is shown only for groups with n >= 20 and both classes present.\n")

    # 9 ---------------------------------------------------------------
    L.append("## 9. Error analysis and explainability\n")
    e = a["errors"]
    L.append(f"At threshold 0.5 the OOF predictions contain {e['false_negatives']} false negatives "
             f"and {e['false_positives']} false positives out of {n} records "
             f"(TP={e['true_positives']}, TN={e['true_negatives']}). The most confident errors:\n")
    bad = pd.read_csv(config.ERRORS_CSV).head(5)
    L.append(md_table(bad[["record_id", "error_type", "p_disease", "age", "sex", "cp",
                           "thalach", "oldpeak", "ca", "thal"]]) + "\n")
    L.append("**Global feature ranking (mean |SHAP|, deployed model)**\n")
    rk = pd.DataFrame(a["shap_ranking"])
    rk["feature"] = rk["feature"].map(config.FEATURE_LABELS)
    L.append(md_table(rk) + "\n")
    for f_, c_ in [("shap_summary.png", "SHAP summary (beeswarm)"),
                   ("shap_bar.png", "Mean absolute SHAP"),
                   ("permutation_importance.png", "Permutation importance")]:
        L.append(fig(f_, c_))

    # 10 --------------------------------------------------------------
    L.append("## 10. Notes on interpretation\n")
    L.append(f"- Sample is small (n={n}); confidence intervals are wide and small model differences "
             "are within noise.")
    L.append("- Data come from one site and an earlier era; no external validation set was used.")
    L.append(f"- Records are {(df.sex == 1).mean():.0%} male and carry no ethnicity information, so "
             "subgroup claims beyond Section 8 cannot be made.")
    L.append("- SHAP describes the model's behaviour, not causal medical relationships.")
    L.append("- Outputs are model estimates for educational use and are not a medical diagnosis.\n")

    # 11 --------------------------------------------------------------
    L.append("## 11. Reproducibility\n")
    v = m["versions"]
    L.append(f"- Seed {config.RANDOM_STATE}; scikit-learn {v['scikit-learn']}, shap {v['shap']}, "
             f"pandas {v['pandas']}, numpy {v['numpy']}.")
    L.append("- Commands: `python -m src.eda && python -m src.train && python -m src.explain "
             "&& python -m src.analysis && python -m src.report`.")

    config.REPORT_MD.write_text("\n".join(x for x in L if x is not None))
    print("Saved", config.REPORT_MD)


if __name__ == "__main__":
    main()
```

**Run:** `python -m src.report`

**Check:**
- `reports/REPORT.md` exists and has 11 numbered sections: `grep -c "^## " reports/REPORT.md` prints `11`.
- No literal `nan` or `None` in the text: `grep -n "nan\|None" reports/REPORT.md` prints nothing (the word `n/a` is allowed).
- Open it in VS Code Markdown preview; figures display.

---

## STEP U5 — Update `src/predict.py` (auto-build all artefacts on a fresh cloud checkout)

**Replace the existing `ensure_model` function** (and only that function) with:

```python
def ensure_model():
    """Build any missing artefact (fresh cloud checkout safety net, ~1-2 min)."""
    if not config.MODEL_PATH.exists():
        from . import explain, train
        train.main()
        explain.main()
    if not config.ANALYSIS_PATH.exists():
        from . import analysis
        analysis.main()
    if not config.REPORT_MD.exists():
        from . import report
        report.main()
```

**Check:** `python -m pytest -q` still passes (3 passed).

---

## STEP U6 — Theme: create `.streamlit/config.toml`

```bash
mkdir -p .streamlit
```

File `.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#0f766e"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f3f6f9"
textColor = "#1f2933"
font = "sans serif"

[server]
headless = true

[browser]
gatherUsageStats = false
```

---

## STEP U7 — Replace `app.py` entirely (the analytics dashboard)

Delete the old content and paste this whole file.

```python
"""CarePredict - analytics dashboard (Streamlit). Educational / research demo."""
import json
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src import config
from src.data import coerce_features, load_data
from src.predict import explain_one, load_bundle, predict_one, risk_band
from src.utils import metrics_at

st.set_page_config(page_title="CarePredict - Analytics", page_icon="🩺", layout="wide")
alt.data_transformers.disable_max_rows()
st.markdown(
    """<style>
    .block-container {padding-top: 1.4rem;}
    [data-testid="stMetric"] {background:#f3f6f9; border:1px solid #e1e6ec;
        border-radius:10px; padding:12px 14px;}
    </style>""",
    unsafe_allow_html=True,
)

OUTCOME = {0: "No disease", 1: "Disease"}
RANGES = {"age": (20, 80), "trestbps": (80, 220), "chol": (100, 600),
          "thalach": (60, 220), "oldpeak": (0.0, 7.0)}


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading model and analysis artefacts...")
def get_bundle():
    return load_bundle()  # also builds any missing artefacts


@st.cache_data
def get_data():
    d = load_data()
    d["Outcome"] = d[config.TARGET].map(OUTCOME)
    return d


@st.cache_data
def read_json(path):
    with open(path) as fh:
        return json.load(fh)


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


pipe, explainer = get_bundle()
data = get_data()
M = read_json(str(config.METRICS_PATH))
A = read_json(str(config.ANALYSIS_PATH))
CMP = read_csv(str(config.COMPARISON_PATH))
OOF = read_csv(str(config.OOF_CSV))
SHAP = read_csv(str(config.SHAP_CSV))
TM = M["test_metrics"]

# ------------------------------------------------------------------ session defaults / examples
DEFAULTS = dict(age=54, sex=1, cp=3, trestbps=130, chol=240, fbs=0, restecg=0,
                thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=3)
EXAMPLES = {
    "low": dict(age=41, sex=0, cp=2, trestbps=130, chol=204, fbs=0, restecg=2,
                thalach=172, exang=0, oldpeak=1.4, slope=1, ca=0, thal=3),
    "high": dict(age=67, sex=1, cp=4, trestbps=160, chol=286, fbs=0, restecg=2,
                 thalach=108, exang=1, oldpeak=1.5, slope=2, ca=3, thal=3),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example(name):
    for k, v in EXAMPLES[name].items():
        st.session_state[k] = v


# ------------------------------------------------------------------ header
st.title("🩺 CarePredict")
st.caption("Explainable machine-learning analytics on the UCI Heart Disease (Cleveland) dataset "
           "- scikit-learn - SHAP - Streamlit")

tabs = st.tabs(["📊 Overview", "🔎 Data explorer", "🧪 Model lab", "🧠 Explainability",
                "🎛️ Patient simulator", "🧬 Subgroups & errors", "📄 Full report"])

# ================================================================== 1. OVERVIEW
with tabs[0]:
    k = st.columns(6)
    k[0].metric("Records", len(data))
    k[1].metric("Features", len(config.FEATURES))
    k[2].metric("Disease prevalence", f"{data[config.TARGET].mean():.1%}")
    k[3].metric("Selected model", M["selected_model"].replace("_", " ").title())
    k[4].metric("Hold-out ROC-AUC", f"{TM['roc_auc']:.3f}",
                help=f"95% bootstrap CI {TM['roc_auc_ci95'][0]}-{TM['roc_auc_ci95'][1]}, n={M['n_test']}")
    k[5].metric("OOF ROC-AUC (all 303)", f"{A['oof']['roc_auc']:.3f}",
                help=f"95% bootstrap CI {A['oof']['roc_auc_ci95'][0]}-{A['oof']['roc_auc_ci95'][1]}")

    st.subheader("Pipeline")
    st.graphviz_chart(
        """digraph { rankdir=LR; node [shape=box, style="rounded,filled",
        fillcolor="#e6f4f1", fontname="Helvetica", fontsize=11];
        A [label="UCI Cleveland\\n303 records"]; B [label="Impute\\n(median / mode)"];
        C [label="One-hot encode\\n+ scale (linear only)"]; D [label="Stratified 5-fold CV\\n4 models"];
        E [label="Best tree model\\nhold-out test"]; F [label="SHAP +\\npermutation"];
        G [label="Dashboard\\n+ report"]; A->B->C->D->E->F->G; }"""
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Model comparison - CV ROC-AUC (mean ± SD)")
        c = CMP.copy()
        c["lo"] = c["roc_auc_mean"] - c["roc_auc_std"]
        c["hi"] = c["roc_auc_mean"] + c["roc_auc_std"]
        bars = alt.Chart(c).mark_bar().encode(
            x=alt.X("model:N", sort="-y", title=None),
            y=alt.Y("roc_auc_mean:Q", scale=alt.Scale(domain=[0, 1]), title="ROC-AUC"),
            color=alt.Color("model:N", legend=None),
            tooltip=["model", "roc_auc_mean", "roc_auc_std", "recall_mean", "precision_mean"])
        rules = alt.Chart(c).mark_rule(color="black").encode(x="model:N", y="lo:Q", y2="hi:Q")
        st.altair_chart((bars + rules).properties(height=300), use_container_width=True)
    with c2:
        st.subheader("Key findings")
        top3 = ", ".join(config.FEATURE_LABELS[r["feature"]] for r in A["shap_ranking"][:3])
        d5 = A["operating_points"]["default_0.5"]
        lr = CMP[CMP.model == "logistic_regression"].iloc[0]
        sel = CMP[CMP.model == M["selected_model"]].iloc[0]
        st.markdown(
            f"- **Best tree model** `{M['selected_model']}`: CV AUC **{sel.roc_auc_mean:.3f}** "
            f"vs logistic regression **{lr.roc_auc_mean:.3f}** and dummy "
            f"**{CMP[CMP.model == 'dummy_most_frequent'].iloc[0].roc_auc_mean:.3f}**.\n"
            f"- **Top drivers (mean |SHAP|):** {top3}.\n"
            f"- **At threshold 0.5 (OOF):** recall **{d5['recall']}**, precision "
            f"**{d5['precision']}**, specificity **{d5['specificity']}**.\n"
            f"- **Calibration:** Brier score **{A['oof']['brier']}**; "
            f"average precision **{A['oof']['avg_precision']}**."
        )
        st.dataframe(CMP[["model", "roc_auc_mean", "roc_auc_std", "recall_mean",
                          "precision_mean", "f1_mean"]], hide_index=True,
                     use_container_width=True)

# ================================================================== 2. DATA EXPLORER
with tabs[1]:
    f1, f2 = st.columns(2)
    sex_sel = f1.multiselect("Sex", [0, 1], default=[0, 1],
                             format_func=lambda v: config.OPTION_LABELS["sex"][v])
    amin, amax = int(data.age.min()), int(data.age.max())
    age_rng = f2.slider("Age range", amin, amax, (amin, amax))
    d = data[data.sex.isin(sex_sel) & data.age.between(*age_rng)]

    if d.empty:
        st.info("No records match the filters.")
    else:
        q = st.columns(3)
        q[0].metric("Filtered records", len(d))
        q[1].metric("Disease prevalence", f"{d[config.TARGET].mean():.1%}")
        q[2].metric("Mean age", f"{d.age.mean():.1f}")

        c1, c2 = st.columns(2)
        with c1:
            col = st.selectbox("Numeric feature distribution", config.NUMERIC,
                               format_func=lambda f: config.FEATURE_LABELS[f])
            hist = alt.Chart(d).mark_bar(opacity=0.65).encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=25), title=config.FEATURE_LABELS[col]),
                y=alt.Y("count():Q", stack=None, title="Records"),
                color=alt.Color("Outcome:N", scale=alt.Scale(
                    domain=["No disease", "Disease"], range=["#1f77b4", "#d62728"])))
            st.altair_chart(hist.properties(height=300), use_container_width=True)
        with c2:
            ccol = st.selectbox("Disease rate by category", config.CATEGORICAL,
                                format_func=lambda f: config.FEATURE_LABELS[f])
            rate = d.groupby(ccol)[config.TARGET].agg(rate="mean", n="size").reset_index()
            rate["level"] = rate[ccol].map(lambda v: config.display_value(ccol, v))
            bar = alt.Chart(rate).mark_bar(color="#0f766e").encode(
                x=alt.X("level:N", title=None, sort=None),
                y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title="Disease rate"),
                tooltip=["level", alt.Tooltip("rate:Q", format=".2f"), "n"])
            st.altair_chart(bar.properties(height=300), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Spearman correlation (numeric + outcome)**")
            corr = d[config.NUMERIC + [config.TARGET]].corr(method="spearman")
            cl = corr.reset_index().melt("index")
            cl.columns = ["a", "b", "corr"]
            base = alt.Chart(cl).encode(x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None))
            heat = base.mark_rect().encode(
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])))
            txt = base.mark_text(fontSize=11).encode(text=alt.Text("corr:Q", format=".2f"))
            st.altair_chart((heat + txt).properties(height=300), use_container_width=True)
        with c4:
            st.markdown("**Missing values per feature**")
            ms = d[config.FEATURES].isna().sum().reset_index()
            ms.columns = ["feature", "missing"]
            st.altair_chart(
                alt.Chart(ms).mark_bar(color="#e76f51").encode(
                    x=alt.X("feature:N", sort="-y", title=None), y="missing:Q",
                    tooltip=["feature", "missing"]).properties(height=300),
                use_container_width=True)

        with st.expander("Raw data"):
            st.dataframe(d.drop(columns=["Outcome"]), use_container_width=True)

# ================================================================== 3. MODEL LAB
with tabs[2]:
    y_oof = OOF["target"].values
    p_sel = OOF["p_selected"].values

    st.subheader("Comparison table")
    cmp_show = CMP.copy()
    cmp_show["OOF_AUC_all_303"] = [A["per_model_oof"][m_]["roc_auc"] for m_ in cmp_show["model"]]
    st.dataframe(cmp_show, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC curves - out-of-fold, all models**")
        parts = []
        for col in [c for c in OOF.columns if c.startswith("p_") and c != "p_selected"]:
            fpr, tpr, _ = roc_curve(y_oof, OOF[col])
            auc = roc_auc_score(y_oof, OOF[col])
            parts.append(pd.DataFrame({"fpr": fpr, "tpr": tpr,
                                       "model": f"{col[2:]} (AUC {auc:.3f})"}))
        roc_df = pd.concat(parts)
        line = alt.Chart(roc_df).mark_line().encode(
            x=alt.X("fpr:Q", title="False positive rate"),
            y=alt.Y("tpr:Q", title="True positive rate"), color="model:N")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
        st.altair_chart((line + diag).properties(height=340), use_container_width=True)
    with c2:
        st.markdown(f"**Precision-recall - {M['selected_model']} (OOF)**")
        pr, rc, _ = precision_recall_curve(y_oof, p_sel)
        prdf = pd.DataFrame({"recall": rc, "precision": pr})
        prl = alt.Chart(prdf).mark_line(color="#e76f51").encode(x="recall:Q", y="precision:Q")
        base_rule = alt.Chart(pd.DataFrame({"y": [y_oof.mean()]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((prl + base_rule).properties(height=340), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Calibration (OOF, quantile bins)**")
        cal = pd.DataFrame(A["calibration"])
        pts = alt.Chart(cal).mark_line(point=True, color="#264653").encode(
            x=alt.X("prob_pred:Q", scale=alt.Scale(domain=[0, 1]), title="Mean predicted probability"),
            y=alt.Y("prob_true:Q", scale=alt.Scale(domain=[0, 1]), title="Observed fraction"))
        dg = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
        st.altair_chart((pts + dg).properties(height=300), use_container_width=True)
    with c4:
        st.markdown("**Score separation by true class (OOF)**")
        sd = pd.DataFrame({"p": p_sel, "Outcome": [OUTCOME[v] for v in y_oof]})
        st.altair_chart(
            alt.Chart(sd).mark_bar(opacity=0.65).encode(
                x=alt.X("p:Q", bin=alt.Bin(maxbins=20), title="Predicted probability"),
                y=alt.Y("count():Q", stack=None),
                color=alt.Color("Outcome:N", scale=alt.Scale(
                    domain=["No disease", "Disease"], range=["#1f77b4", "#d62728"])),
            ).properties(height=300), use_container_width=True)

    st.subheader("Threshold explorer")
    t = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    mt = metrics_at(y_oof, p_sel, t)
    tcols = st.columns(5)
    tcols[0].metric("Recall (sensitivity)", mt["recall"])
    tcols[1].metric("Specificity", mt["specificity"])
    tcols[2].metric("Precision", mt["precision"])
    tcols[3].metric("F1", mt["f1"])
    tcols[4].metric("Accuracy", mt["accuracy"])
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        st.markdown("**Confusion matrix (OOF, all 303)**")
        st.dataframe(pd.DataFrame([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]],
                                  index=["Actual: no disease", "Actual: disease"],
                                  columns=["Predicted: no", "Predicted: yes"]),
                     use_container_width=True)
        ops = A["operating_points"]
        st.caption(f"Youden-optimal threshold: {ops['youden']['threshold']} - "
                   + (f"highest threshold with recall >= 0.90: {ops['recall_ge_90']['threshold']}"
                      if ops["recall_ge_90"] else "no threshold reaches recall 0.90"))
    with cc2:
        sweep = pd.DataFrame([metrics_at(y_oof, p_sel, x) for x in np.arange(0.02, 0.99, 0.02)])
        long = sweep.melt("threshold", ["precision", "recall", "specificity", "f1"],
                          var_name="metric", value_name="value")
        lines = alt.Chart(long).mark_line().encode(x="threshold:Q", y="value:Q", color="metric:N")
        vr = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(color="black").encode(x="t:Q")
        st.altair_chart((lines + vr).properties(height=260), use_container_width=True)

# ================================================================== 4. EXPLAINABILITY
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Global importance - mean |SHAP|")
        rk = pd.DataFrame(A["shap_ranking"])
        rk["label"] = rk["feature"].map(config.FEATURE_LABELS)
        st.altair_chart(
            alt.Chart(rk).mark_bar(color="#0f766e").encode(
                x=alt.X("mean_abs_shap:Q", title="Mean |SHAP value|"),
                y=alt.Y("label:N", sort="-x", title=None),
                tooltip=["label", "mean_abs_shap"]).properties(height=380),
            use_container_width=True)
    with c2:
        st.subheader("SHAP summary (beeswarm)")
        pth = config.FIG_DIR / "shap_summary.png"
        if pth.exists():
            st.image(str(pth))

    st.subheader("SHAP dependence plot")
    feat = st.selectbox("Feature", config.FEATURES, format_func=lambda f: config.FEATURE_LABELS[f])
    dep = pd.DataFrame({"value": SHAP[f"val_{feat}"], "shap": SHAP[f"shap_{feat}"],
                        "Outcome": SHAP["target"].map(OUTCOME)}).dropna()
    st.altair_chart(
        alt.Chart(dep).mark_circle(size=70, opacity=0.6).encode(
            x=alt.X("value:Q", title=config.FEATURE_LABELS[feat]),
            y=alt.Y("shap:Q", title="SHAP contribution (positive = higher risk)"),
            color=alt.Color("Outcome:N", scale=alt.Scale(
                domain=["No disease", "Disease"], range=["#1f77b4", "#d62728"])),
            tooltip=["value", "shap", "Outcome"]).properties(height=340).interactive(),
        use_container_width=True)

    pth = config.FIG_DIR / "permutation_importance.png"
    if pth.exists():
        st.subheader("Permutation importance (hold-out test set)")
        st.image(str(pth), width=620)

# ================================================================== 5. PATIENT SIMULATOR
with tabs[4]:
    left, right = st.columns([1, 1.5])

    def num(container, f, **kw):
        lo, hi = RANGES[f]
        return container.slider(config.FEATURE_LABELS[f], lo, hi, key=f, **kw)

    def sel(container, f):
        return container.selectbox(
            config.FEATURE_LABELS[f], list(config.OPTION_LABELS[f]), key=f,
            format_func=lambda v, ff=f: config.OPTION_LABELS[ff][v])

    with left:
        st.subheader("Patient features")
        b1, b2 = st.columns(2)
        b1.button("Load lower-risk example", on_click=load_example, args=("low",))
        b2.button("Load higher-risk example", on_click=load_example, args=("high",))
        row = {}
        row["age"] = num(st, "age")
        row["sex"] = sel(st, "sex")
        row["cp"] = sel(st, "cp")
        row["trestbps"] = num(st, "trestbps")
        row["chol"] = num(st, "chol")
        row["fbs"] = sel(st, "fbs")
        row["restecg"] = sel(st, "restecg")
        row["thalach"] = num(st, "thalach")
        row["exang"] = sel(st, "exang")
        row["oldpeak"] = num(st, "oldpeak", step=0.1)
        row["slope"] = sel(st, "slope")
        row["ca"] = sel(st, "ca")
        row["thal"] = sel(st, "thal")

    with right:
        p = predict_one(pipe, row)
        pct = float((OOF["p_selected"] < p).mean())
        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated probability", f"{p:.0%}")
        r2.metric("Band", risk_band(p).replace(" estimated risk", ""))
        r3.metric("Percentile in dataset", f"{pct:.0%}",
                  help="Share of the 303 out-of-fold predictions that are lower than this one")
        st.progress(min(max(p, 0.0), 1.0))

        st.markdown("**Why this estimate - SHAP contributions**")
        contrib = explain_one(pipe, explainer, row)
        cdf = pd.DataFrame({"feature": contrib.index, "contribution": contrib.values})
        cdf["label"] = [f"{config.FEATURE_LABELS[f]} = {config.display_value(f, row[f])}"
                        for f in cdf.feature]
        cdf["abs"] = cdf["contribution"].abs()
        st.altair_chart(
            alt.Chart(cdf.head(8)).mark_bar().encode(
                x=alt.X("contribution:Q", title="SHAP contribution (right = higher risk)"),
                y=alt.Y("label:N", sort=alt.EncodingSortField(field="abs", order="descending"),
                        title=None),
                color=alt.condition(alt.datum.contribution > 0,
                                    alt.value("#d62728"), alt.value("#1f77b4")),
                tooltip=["label", "contribution"]).properties(height=280),
            use_container_width=True)

        st.markdown("**What-if analysis - vary one feature, hold the rest fixed**")
        sweep_f = st.selectbox("Feature to vary", config.NUMERIC,
                               format_func=lambda f: config.FEATURE_LABELS[f])
        lo, hi = RANGES[sweep_f]
        grid = np.linspace(lo, hi, 40)
        sweep_df = pd.DataFrame([dict(row, **{sweep_f: v}) for v in grid])
        probs = pipe.predict_proba(coerce_features(sweep_df))[:, 1]
        wf = pd.DataFrame({"x": grid, "probability": probs})
        curve = alt.Chart(wf).mark_line(color="#0f766e").encode(
            x=alt.X("x:Q", title=config.FEATURE_LABELS[sweep_f]),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])))
        here = alt.Chart(pd.DataFrame({"x": [row[sweep_f]], "probability": [p]})).mark_point(
            color="#d62728", size=140, filled=True).encode(x="x:Q", y="probability:Q")
        st.altair_chart((curve + here).properties(height=240), use_container_width=True)

# ================================================================== 6. SUBGROUPS & ERRORS
with tabs[5]:
    st.subheader("Performance by subgroup (OOF, threshold 0.5)")
    sub = pd.DataFrame(A["subgroups"])
    st.dataframe(sub, hide_index=True, use_container_width=True)
    metric = st.selectbox("Metric to chart", ["recall", "precision", "accuracy", "roc_auc"])
    st.altair_chart(
        alt.Chart(sub).mark_bar(color="#0f766e").encode(
            x=alt.X("group:N", title=None, sort=None),
            y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 1])),
            column=alt.Column("attribute:N", title=None),
            tooltip=["attribute", "group", "n", metric]).resolve_scale(x="independent"))

    st.subheader("Error analysis (OOF, threshold 0.5)")
    e = A["errors"]
    ec = st.columns(4)
    ec[0].metric("True positives", e["true_positives"])
    ec[1].metric("True negatives", e["true_negatives"])
    ec[2].metric("False positives", e["false_positives"])
    ec[3].metric("False negatives", e["false_negatives"])
    errs = read_csv(str(config.ERRORS_CSV))
    kind = st.radio("Show", ["All errors", "False negatives", "False positives"], horizontal=True)
    if kind != "All errors":
        errs = errs[errs["error_type"] == kind.rstrip("s")]
    st.dataframe(errs, hide_index=True, use_container_width=True)

# ================================================================== 7. FULL REPORT
with tabs[6]:
    if config.REPORT_MD.exists():
        text = config.REPORT_MD.read_text()
        st.download_button("⬇️ Download REPORT.md", data=text,
                           file_name="CarePredict_REPORT.md", mime="text/markdown")
        pos = 0
        for mt_ in re.finditer(r"!\[(.*?)\]\((.*?)\)", text):
            st.markdown(text[pos:mt_.start()])
            img = config.REPORT_DIR / mt_.group(2)
            if img.exists():
                st.image(str(img), caption=mt_.group(1), width=640)
            pos = mt_.end()
        st.markdown(text[pos:])
    else:
        st.info("Run `python -m src.report` to generate the report.")

st.divider()
st.caption("CarePredict - research & education demo - model outputs are not a medical diagnosis.")
```

**Run:** `streamlit run app.py`

### Click-test checklist (every item must pass)
1. No warning banner at the top; no "Limitations" tab exists.
2. **Overview:** six KPI cards, pipeline diagram, bar chart with error whiskers, "Key findings" bullets with real numbers.
3. **Data explorer:** change the Sex filter / age range → all counters and charts update; "Raw data" expander opens.
4. **Model lab:** ROC overlay shows 4 lines; moving the threshold slider changes the metrics, the 2×2 confusion table, and the vertical line.
5. **Explainability:** pick different features in the dependence plot → scatter changes.
6. **Patient simulator:** both example buttons change inputs, probability, percentile, and SHAP bars; the what-if curve has a red dot at the current value.
7. **Subgroups & errors:** table, faceted bar chart, error list with the radio filter.
8. **Full report:** download button works; figures appear inline.
9. Terminal shows no tracebacks.

---

## STEP U8 — Add tests

Create `tests/test_analysis.py`:

```python
import json

from src import config
from src.predict import ensure_model


def test_artifacts_exist_and_consistent():
    ensure_model()
    for path in (config.ANALYSIS_PATH, config.OOF_CSV, config.SHAP_CSV,
                 config.ERRORS_CSV, config.REPORT_MD):
        assert path.exists(), path
    a = json.loads(config.ANALYSIS_PATH.read_text())
    assert 0.5 < a["oof"]["roc_auc"] <= 1.0
    lo, hi = a["oof"]["roc_auc_ci95"]
    assert lo <= a["oof"]["roc_auc"] <= hi
    e = a["errors"]
    assert e["true_positives"] + e["true_negatives"] + e["false_positives"] + e["false_negatives"] == a["n_records"]


def test_report_has_all_sections():
    ensure_model()
    text = config.REPORT_MD.read_text()
    assert text.count("\n## ") + text.startswith("## ") == 11
    assert "nan" not in text.lower().replace("finance", "")
```

**Run:** `python -m pytest -q` → `5 passed`.

> If the `nan` assertion fails, open `reports/REPORT.md`, find the offending table cell, and see Section 12 (R2).

---

## STEP U9 — Final regeneration, commit, redeploy

```bash
python -m src.eda && python -m src.train && python -m src.explain && python -m src.analysis && python -m src.report
python -m pytest -q
pip freeze | grep -iE "^(numpy|pandas|scikit-learn|shap|matplotlib|streamlit|joblib)=="   # must equal requirements.txt pins
git add .
git commit -m "Analytics dashboard + full results report"
git push
```

Streamlit Community Cloud redeploys automatically on push (1–3 min). If the page shows the old version, open the app's menu → **Reboot app**.

**Commit these artefacts** (the app reads them): `models/model.joblib`, `models/metrics.json`, `models/analysis.json`, `reports/*.csv`, `reports/REPORT.md`, `reports/figures/*.png`, `data/`, `.streamlit/config.toml`.

Also update `README.md`: replace the "How to run" commands with the new run order (top of this file), add 2–3 screenshots, and add a "Results" table copied from Section 5/6 of `reports/REPORT.md`.

---

## STEP U10 — Updated 5-minute demo script

| Min | Tab | Say / do |
|---|---|---|
| 0:00 | Overview | "End-to-end pipeline on public UCI data. 303 records, 4 models, stratified CV. Best tree model reaches AUC ≈ *(read the KPI card)*, baseline dummy 0.5." |
| 1:00 | Data explorer | Filter sex = Male and age ≥ 55; show prevalence changes; pick `thalach` histogram and `cp` disease-rate chart. |
| 2:00 | Model lab | ROC overlay; drag the threshold slider from 0.5 to 0.3: "recall goes up, precision falls — this is the screening trade-off." Mention Youden and recall≥0.90 thresholds. |
| 3:00 | Explainability | Mean |SHAP| ranking; open the dependence plot for `oldpeak` or `age`. |
| 3:45 | Patient simulator | Load higher-risk example → show SHAP bars; use the what-if curve on `thalach`. Load lower-risk example for contrast. |
| 4:30 | Subgroups & errors | Show recall by sex/age band and the most confident errors. |
| 5:00 | Full report | Click download; scroll the executive summary. |

---

## 12. TROUBLESHOOTING (additions)

| # | Symptom | Fix |
|---|---|---|
| R1 | `ModuleNotFoundError: src.utils` / `src.analysis` | File not created or wrong folder; both must be in `src/`. |
| R2 | `nan`/`None` appears in `REPORT.md` | The text comes from a pandas cell; `md_table` already prints `n/a` for NaN. If it is inside a sentence, print the variable it uses and check the source JSON. |
| R3 | `KeyError: 'logistic_regression'` in `report.py` | `reports/model_comparison.csv` is stale or edited. Re-run `python -m src.train`, then `python -m src.analysis`, then `python -m src.report`. |
| R4 | `analysis.py` slow (>2 min) | Normal on slow CPUs (it cross-validates 4 models). Wait; do not reduce folds. |
| R5 | Altair error `Javascript Error` / blank chart | Column has `NaN` in a field used as `:N`. Check the data passed to the chart with `print(df.head())`. Dependence plot already uses `.dropna()`. |
| R6 | `StreamlitAPIException: ... value ... session state` | Widgets that use `key=` must not receive `value=`/`index=`. Use the code exactly as written. |
| R7 | Graphviz chart does not render | Safe to skip; replace the `st.graphviz_chart(...)` block with `st.markdown("Data → Impute → Encode → CV (4 models) → Hold-out → SHAP → Dashboard")`. |
| R8 | Error-type radio shows no rows | In the old code snippet, the filter maps "False negatives"→"False negative"; keep that line exactly. |
| R9 | Cloud app first load takes minutes | Artefacts missing from the repo, so the app is rebuilding them. Commit the files listed in U9. |
| R10 | Figures missing in the Full report tab | Run the whole run order again, then `git add reports && git push`. |

---

## 13. FINAL ACCEPTANCE CHECKLIST

- [ ] New run order executes end to end without errors.
- [ ] `python -m pytest -q` → 5 passed.
- [ ] `reports/REPORT.md` has 11 sections, tables, and embedded figures; no `nan`.
- [ ] Dashboard has 7 tabs, **no warning banner and no limitations tab**, only the one-line footer.
- [ ] Threshold slider, filters, dependence plot, simulator and what-if curve all react to input.
- [ ] Report downloads from the app.
- [ ] Changes pushed; the public Streamlit URL shows the new dashboard (incognito check).
- [ ] README updated with new run order and results table.

**End of agent_v2_analytics_upgrade.md**