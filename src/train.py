"""Train + compare models, evaluate on hold-out test set, save artefacts."""
import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data import coerce_features, load_data
from .features import build_preprocessor


def make_candidates():
    rs = config.RANDOM_STATE
    return {
        "dummy_most_frequent": dict(
            deployable=False,
            pipe=Pipeline([
                ("prep", build_preprocessor(scale=False)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]),
        ),
        "logistic_regression": dict(
            deployable=False,  # baseline; not a tree model
            pipe=Pipeline([
                ("prep", build_preprocessor(scale=True)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=rs)),
            ]),
        ),
        "random_forest": dict(
            deployable=True,
            pipe=Pipeline([
                ("prep", build_preprocessor(scale=False)),
                ("model", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                    class_weight="balanced_subsample", random_state=rs, n_jobs=1)),
            ]),
        ),
        "gradient_boosting": dict(
            deployable=True,
            pipe=Pipeline([
                ("prep", build_preprocessor(scale=False)),
                ("model", GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.05, max_depth=2,
                    subsample=0.8, random_state=rs)),
            ]),
        ),
    }


def main():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = coerce_features(df)
    y = df[config.TARGET]

    # ---- 1. stratified hold-out split (test set is touched only once, at the end) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    test_df = X_test.copy()
    test_df[config.TARGET] = y_test
    test_df.to_csv(config.TEST_CSV, index=False)

    # ---- 2. stratified 5-fold CV on the training part only ----
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy",
               "precision": "precision", "recall": "recall", "f1": "f1"}
    candidates = make_candidates()
    rows = []
    for name, spec in candidates.items():
        res = cross_validate(spec["pipe"], X_train, y_train, cv=cv, scoring=scoring,
                             n_jobs=1, error_score=0.0)
        row = {"model": name, "deployable_tree_model": spec["deployable"]}
        for m in scoring:
            vals = res[f"test_{m}"]
            row[f"{m}_mean"] = round(float(np.mean(vals)), 3)
            row[f"{m}_std"] = round(float(np.std(vals)), 3)
        rows.append(row)
        print(f"{name:22s} CV ROC-AUC = {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f}")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.COMPARISON_PATH, index=False)

    # ---- 3. pick best TREE model (SHAP TreeExplainer requires a tree model) ----
    tree_rows = comparison[comparison["deployable_tree_model"]]
    best_name = tree_rows.sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
    print("\nSelected model:", best_name)
    best_pipe = candidates[best_name]["pipe"]
    best_pipe.fit(X_train, y_train)

    # ---- 4. evaluate ONCE on hold-out test set ----
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_arr = y_test.values
    test_metrics = {
        "roc_auc": round(float(roc_auc_score(y_arr, proba)), 3),
        "accuracy": round(float(accuracy_score(y_arr, pred)), 3),
        "precision": round(float(precision_score(y_arr, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_arr, pred)), 3),
        "f1": round(float(f1_score(y_arr, pred)), 3),
    }
    # bootstrap 95% CI for test AUC (test set is small -> CI is wide; report it honestly)
    rng = np.random.default_rng(config.RANDOM_STATE)
    aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(set(y_arr[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_arr[idx], proba[idx]))
    ci = np.percentile(aucs, [2.5, 97.5])
    test_metrics["roc_auc_ci95"] = [round(float(ci[0]), 3), round(float(ci[1]), 3)]
    print("Test metrics:", test_metrics)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_arr, proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC curve (hold-out test set)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_arr, pred, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ---- 5. save ----
    joblib.dump(best_pipe, config.MODEL_PATH)
    metrics = {
        "selected_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_positive_rate": round(float(y_train.mean()), 3),
        "test_positive_rate": round(float(y_test.mean()), 3),
        "cv_table": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "versions": {"scikit-learn": sklearn.__version__, "shap": shap.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("\nSaved:", config.MODEL_PATH, config.METRICS_PATH, config.COMPARISON_PATH)


if __name__ == "__main__":
    main()
