"""CarePredict - minimal Streamlit demo. NOT a medical device."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.predict import explain_one, load_bundle, predict_one, risk_band

st.set_page_config(page_title="CarePredict", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_bundle():
    return load_bundle()


pipe, explainer = get_bundle()

# ------------------------------------------------------------------ state / examples
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
st.title("🩺 CarePredict — Heart-Disease Risk Estimator")
st.warning(
    "**Educational demo only — NOT a medical diagnosis.** Built on a small, old, "
    "single-hospital public dataset (UCI Cleveland, 303 patients). "
    "Do not use it to make health decisions. Consult a qualified clinician."
)

# ------------------------------------------------------------------ sidebar inputs
st.sidebar.header("Patient features")
st.sidebar.button("Load example: lower-risk profile", on_click=load_example, args=("low",))
st.sidebar.button("Load example: higher-risk profile", on_click=load_example, args=("high",))


def select(feature):
    opts = list(config.OPTION_LABELS[feature].keys())
    return st.sidebar.selectbox(
        config.FEATURE_LABELS[feature], opts, key=feature,
        format_func=lambda v, f=feature: config.OPTION_LABELS[f][v],
    )


row = {}
row["age"] = st.sidebar.slider(config.FEATURE_LABELS["age"], 20, 80, key="age")
row["sex"] = select("sex")
row["cp"] = select("cp")
row["trestbps"] = st.sidebar.slider(config.FEATURE_LABELS["trestbps"], 80, 220, key="trestbps")
row["chol"] = st.sidebar.slider(config.FEATURE_LABELS["chol"], 100, 600, key="chol")
row["fbs"] = select("fbs")
row["restecg"] = select("restecg")
row["thalach"] = st.sidebar.slider(config.FEATURE_LABELS["thalach"], 60, 220, key="thalach")
row["exang"] = select("exang")
row["oldpeak"] = st.sidebar.slider(config.FEATURE_LABELS["oldpeak"], 0.0, 7.0, step=0.1, key="oldpeak")
row["slope"] = select("slope")
row["ca"] = select("ca")
row["thal"] = select("thal")

# ------------------------------------------------------------------ tabs
tab_pred, tab_perf, tab_global, tab_limits = st.tabs(
    ["Prediction", "Model performance", "Global explanations", "Limitations & bias"]
)

with tab_pred:
    p = predict_one(pipe, row)
    band = risk_band(p)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Estimated probability (model output)", f"{p:.0%}")
        st.progress(min(max(p, 0.0), 1.0))
        if p < 0.33:
            st.success(band)
        elif p < 0.66:
            st.warning(band)
        else:
            st.error(band)
        st.caption("Bands are illustrative, not clinical thresholds.")
    with c2:
        st.subheader("Why this estimate? (SHAP)")
        contrib = explain_one(pipe, explainer, row)
        top = contrib.head(8)[::-1]
        labels = [f"{config.FEATURE_LABELS[f]} = {config.display_value(f, row[f])}"
                  for f in top.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(labels, top.values,
                color=["#d62728" if v > 0 else "#1f77b4" for v in top.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("SHAP contribution (red/right = pushes risk UP, blue/left = DOWN)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Explains the model's behaviour, not medical cause and effect.")

with tab_perf:
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        st.write(f"**Deployed model:** `{m['selected_model']}` "
                 f"(train n={m['n_train']}, test n={m['n_test']})")
        st.subheader("Stratified 5-fold CV on training data (baseline vs tree models)")
        st.dataframe(pd.read_csv(config.COMPARISON_PATH), use_container_width=True)
        st.subheader("Hold-out test set (used once)")
        st.json(m["test_metrics"])
        st.caption("The test set is small, so the confidence interval is wide.")
        c1, c2 = st.columns(2)
        for col, img in zip((c1, c2), ("roc_curve.png", "confusion_matrix.png")):
            path = config.FIG_DIR / img
            if path.exists():
                col.image(str(path))
    else:
        st.info("Run `python -m src.train` first.")

with tab_global:
    for title, img in [("SHAP summary (all records)", "shap_summary.png"),
                       ("Mean |SHAP| importance", "shap_bar.png"),
                       ("Permutation importance (test set)", "permutation_importance.png")]:
        path = config.FIG_DIR / img
        if path.exists():
            st.subheader(title)
            st.image(str(path))

with tab_limits:
    if config.MODEL_CARD_PATH.exists():
        st.markdown(config.MODEL_CARD_PATH.read_text())
    else:
        st.info("MODEL_CARD.md not found.")
