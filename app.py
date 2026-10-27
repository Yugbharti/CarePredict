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
