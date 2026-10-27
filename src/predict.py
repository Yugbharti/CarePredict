"""Load the model and make single-patient predictions with SHAP explanations."""
import joblib
import pandas as pd
import shap

from . import config
from .data import coerce_features
from .explain import original_feature, positive_class_shap, transform_df

LOW_MAX = 0.33     # illustrative bands, NOT clinical thresholds
HIGH_MIN = 0.66


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


def load_bundle():
    """Return (pipeline, shap_explainer)."""
    ensure_model()
    pipe = joblib.load(config.MODEL_PATH)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return pipe, explainer


def _row_df(row: dict) -> pd.DataFrame:
    return coerce_features(pd.DataFrame([row]))


def predict_one(pipe, row: dict) -> float:
    """Probability (0-1) of the positive class for one patient dict."""
    return float(pipe.predict_proba(_row_df(row))[0, 1])


def risk_band(p: float) -> str:
    if p < LOW_MAX:
        return "Lower estimated risk"
    if p < HIGH_MIN:
        return "Intermediate estimated risk"
    return "Higher estimated risk"


def explain_one(pipe, explainer, row: dict) -> pd.Series:
    """SHAP contribution per ORIGINAL feature (one-hot columns are summed back).

    Positive value = pushes risk up. Sorted by absolute size, largest first.
    """
    X_t = transform_df(pipe, _row_df(row))
    sv = positive_class_shap(explainer, X_t.values)[0]
    grouped = {}
    for name, val in zip(X_t.columns, sv):
        f = original_feature(name)
        grouped[f] = grouped.get(f, 0.0) + float(val)
    s = pd.Series(grouped)
    return s.sort_values(key=lambda v: v.abs(), ascending=False)
