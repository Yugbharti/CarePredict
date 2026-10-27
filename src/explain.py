"""Permutation importance, SHAP global plots, and SHAP helpers for the app."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from . import config
from .data import coerce_features, load_data


def transform_df(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply only the preprocessing step; return a DataFrame with clean column names."""
    prep = pipe.named_steps["prep"]
    arr = prep.transform(X)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    return pd.DataFrame(arr, columns=names, index=X.index)


def original_feature(transformed_name: str) -> str:
    """'cp_4.0' -> 'cp', 'age' -> 'age'."""
    for f in config.FEATURES:
        if transformed_name == f or transformed_name.startswith(f + "_"):
            return f
    return transformed_name


def positive_class_shap(explainer, X) -> np.ndarray:
    """SHAP values for the 'disease' class as a 2-D array, across SHAP versions."""
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):          # older SHAP: list of arrays per class
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer SHAP: (rows, features, classes)
        sv = sv[:, :, -1]
    return sv


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(config.MODEL_PATH)

    # ---- permutation importance on the hold-out test set (original features) ----
    test = pd.read_csv(config.TEST_CSV)
    X_test, y_test = coerce_features(test), test[config.TARGET]
    pi = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                n_repeats=30, random_state=config.RANDOM_STATE)
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(config.FEATURES)[order], pi.importances_mean[order],
            xerr=pi.importances_std[order])
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Permutation importance (hold-out test set)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "permutation_importance.png", dpi=120)
    plt.close(fig)

    # ---- SHAP global plots (computed on all 303 records) ----
    df = load_data()
    X_all = coerce_features(df)
    X_t = transform_df(pipe, X_all)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = positive_class_shap(explainer, X_t.values)

    plt.figure()
    shap.summary_plot(sv, X_t, show=False)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_t, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("Saved permutation_importance.png, shap_summary.png, shap_bar.png")


if __name__ == "__main__":
    main()
