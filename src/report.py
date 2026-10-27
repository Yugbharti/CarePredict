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
