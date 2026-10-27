# agent.md — CarePredict: Healthcare Risk Prediction (Implementation Guide)

> **Audience:** an AI coding agent (including a small/low-end model). Follow this file literally, phase by phase.
> **Goal:** a working ML pipeline + SHAP explanations + a minimal Streamlit app that can be **demonstrated live** and **deployed free on the cloud**.
> **Project one-liner:** *CarePredict estimates the probability that a patient has heart disease from 13 clinical features, explains each prediction with SHAP, and clearly states it is NOT a medical diagnosis.*

---

## 0. RULES FOR THE AGENT (read first, obey always)

1. **Do phases in order.** Do not start Phase N+1 until every "DONE WHEN" check of Phase N passes.
2. **Copy the code blocks exactly.** Do not rename files, folders, functions, or variables. Other files import them by name.
3. **Run every command from the repo root** (the folder that contains `app.py`). Run modules as `python -m src.xxx` (not `python src/xxx.py`).
4. **If something fails, go to Section 14 (Troubleshooting) before inventing a fix.** Never delete a check to make it pass.
5. **Do not change the dataset, target, or train/test split logic.** Do not tune on the test set.
6. **Never write text claiming the model diagnoses disease.** Always keep the disclaimer visible in the app.
7. **Do not invent numbers.** Any number in `MODEL_CARD.md`/README must be copied from `models/metrics.json` or `reports/eda_summary.txt`.
8. Keep it simple. No extra frameworks (no Docker, no databases, no FastAPI, no deep learning).
9. After each phase, print what you did and the result of the checks.

**Estimated total time:** 2–3 hours.

---

## 1. PROJECT OVERVIEW

| Item | Decision |
|---|---|
| Dataset | UCI **Heart Disease (Cleveland subset)** — public, de-identified, 303 rows, 13 features |
| Source URL | `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data` |
| Outcome | `target = 1` if angiographic heart disease is present (original column `num` > 0), else `0` |
| Missing values | Real ones exist (`ca`: 4, `thal`: 2) → imputation is genuinely needed |
| Models compared | Dummy baseline, Logistic Regression (baseline), Random Forest, Gradient Boosting (tree-based) |
| Validation | 80/20 stratified hold-out split + 5-fold **stratified** CV on the training part |
| Deployed model | Best **tree-based** model by CV ROC-AUC (SHAP `TreeExplainer` needs a tree model) |
| Explainability | Permutation feature importance + SHAP global plots + SHAP per-patient bar chart |
| App | Streamlit (single `app.py`) |
| Free hosting | **Streamlit Community Cloud** (primary), Hugging Face Spaces (backup) |

### Resume bullet → where it is satisfied

| Resume claim | Evidence in project |
|---|---|
| Supervised learning pipeline on public de-identified data | Phases 2–5, `src/train.py` |
| Missing-value treatment, categorical encoding, scaling where appropriate, EDA | `src/features.py`, `src/eda.py` (Phases 3–4) |
| Baseline vs tree-based models with stratified validation | `reports/model_comparison.csv` (Phase 5) |
| Feature importance + SHAP | `src/explain.py`, app "Global explanations" tab (Phase 6, 8) |
| Documented bias, generalisation limits, "not a diagnosis" | `MODEL_CARD.md`, app disclaimer (Phases 9, 8) |

---

## 2. FINAL REPOSITORY STRUCTURE (target)

```
carepredict/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── MODEL_CARD.md
├── agent.md                  (this file)
├── .gitignore
├── data/
│   ├── raw/processed.cleveland.data     (downloaded; COMMIT it)
│   └── processed/test.csv               (created by training; COMMIT it)
├── models/
│   ├── model.joblib                     (created by training; COMMIT it)
│   └── metrics.json                     (created by training; COMMIT it)
├── reports/
│   ├── eda_summary.txt
│   ├── model_comparison.csv
│   └── figures/*.png                    (COMMIT them)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── eda.py
│   ├── train.py
│   ├── explain.py
│   └── predict.py
└── tests/
    └── test_smoke.py
```

---

## PHASE 1 — Environment & scaffold (≈10 min)

### 1.1 Prerequisites
- Python **3.10, 3.11 or 3.12** (`python --version`). Prefer 3.11.
- `git` installed, a free GitHub account.

### 1.2 Commands

```bash
mkdir carepredict && cd carepredict
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

mkdir -p data/raw data/processed models reports/figures src tests
touch src/__init__.py
```

### 1.3 Create `requirements.txt`

```
numpy>=1.26,<2.3
pandas>=2.1,<2.4
scikit-learn>=1.4,<1.8
shap>=0.45
matplotlib>=3.8
streamlit>=1.35
joblib>=1.3
```

### 1.4 Create `requirements-dev.txt`

```
-r requirements.txt
pytest>=8.0
```

### 1.5 Create `.gitignore`

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
```

### 1.6 Install

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### DONE WHEN
```bash
python -c "import pandas, sklearn, shap, streamlit, matplotlib, joblib; print('imports ok')"
```
prints `imports ok`.

---

## PHASE 2 — Config & data loading (≈15 min)

### 2.1 Create `src/config.py`

```python
"""Central configuration. Every other module imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- paths ----
RAW_FILE = ROOT / "data" / "raw" / "processed.cleveland.data"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
MODEL_CARD_PATH = ROOT / "MODEL_CARD.md"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# ---- schema ----
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---- human-readable names (used by the app) ----
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak-exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia test result",
}

OPTION_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina",
           3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


def display_value(feature, value):
    """Readable text for a feature value."""
    if feature in OPTION_LABELS:
        try:
            return OPTION_LABELS[feature][int(value)]
        except (KeyError, ValueError, TypeError):
            return str(value)
    return str(value)
```

### 2.2 Create `src/data.py`

```python
"""Download and load the UCI Cleveland heart-disease dataset."""
import urllib.request

import pandas as pd

from . import config


def download_data(force: bool = False):
    """Download the raw file once. Safe to call repeatedly."""
    if config.RAW_FILE.exists() and not force:
        return config.RAW_FILE
    config.RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            config.DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            config.RAW_FILE.write_bytes(resp.read())
    except Exception as exc:  # network problem
        raise RuntimeError(
            "Could not download the dataset. Manually download "
            f"{config.DATA_URL} in a browser and save it as "
            f"{config.RAW_FILE}, then re-run."
        ) from exc
    return config.RAW_FILE


def load_data() -> pd.DataFrame:
    """Return a DataFrame with the 13 features + binary `target`."""
    download_data()
    df = pd.read_csv(
        config.RAW_FILE, header=None, names=config.RAW_COLUMNS, na_values="?"
    )
    df[config.TARGET] = (df["num"] > 0).astype(int)  # 0 = no disease, 1 = disease
    return df.drop(columns="num")


def coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns in the right order and make them all float.

    Used in training AND inference so category values always match
    (e.g. 4 and 4.0 are treated identically).
    """
    return X[config.FEATURES].astype(float)
```

### 2.3 Run

```bash
python -c "from src.data import load_data; d=load_data(); print(d.shape); print(d.isna().sum().sum()); print(d['target'].mean().round(3))"
```

### DONE WHEN
- Shape printed is `(303, 14)`.
- Total missing printed is `6`.
- Positive rate printed is about `0.459`.
- File `data/raw/processed.cleveland.data` exists.

---

## PHASE 3 — Exploratory data analysis (≈15 min)

### 3.1 Create `src/eda.py`

```python
"""EDA: writes reports/eda_summary.txt and several figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .data import load_data


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    lines = [
        f"Rows: {len(df)}   Columns: {df.shape[1]}",
        "\nMissing values per column:\n" + df.isna().sum().to_string(),
        "\nTarget balance (proportion):\n"
        + df[config.TARGET].value_counts(normalize=True).round(3).to_string(),
        "\nSex distribution (1=male, 0=female):\n"
        + df["sex"].value_counts(normalize=True).round(3).to_string(),
        "\nAge summary:\n" + df["age"].describe().round(1).to_string(),
        "\nNumeric feature summary:\n"
        + df[config.NUMERIC].describe().round(2).to_string(),
    ]
    text = "\n".join(lines)
    (config.REPORT_DIR / "eda_summary.txt").write_text(text)
    print(text)

    # 1. target balance
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = df[config.TARGET].value_counts().sort_index()
    ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=["#1f77b4", "#d62728"])
    ax.set_title("Target balance")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_target_balance.png", dpi=120)
    plt.close(fig)

    # 2. missing values
    fig, ax = plt.subplots(figsize=(6, 3))
    miss = df.isna().sum()
    ax.bar(miss.index, miss.values)
    ax.set_title("Missing values per column")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_missing.png", dpi=120)
    plt.close(fig)

    # 3. numeric histograms split by target
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, config.NUMERIC):
        for label, grp in df.groupby(config.TARGET):
            ax.hist(grp[col].dropna(), bins=20, alpha=0.6, label=f"target={label}")
        ax.set_title(col)
        ax.legend()
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_numeric_by_target.png", dpi=120)
    plt.close(fig)

    # 4. positive rate per categorical level
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for ax, col in zip(axes, config.CATEGORICAL):
        rate = df.groupby(col)[config.TARGET].mean()
        ax.bar(rate.index.astype(str), rate.values)
        ax.set_ylim(0, 1)
        ax.set_title(f"Disease rate by {col}")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_categorical_rates.png", dpi=120)
    plt.close(fig)

    # 5. correlation heatmap (numeric + target)
    cols = config.NUMERIC + [config.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60)
    ax.set_yticklabels(cols)
    fig.colorbar(im)
    ax.set_title("Correlation")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_correlation.png", dpi=120)
    plt.close(fig)

    print("\nEDA figures saved to", config.FIG_DIR)


if __name__ == "__main__":
    main()
```

### 3.2 Run
```bash
python -m src.eda
```

### DONE WHEN
- `reports/eda_summary.txt` exists and shows missing values only in `ca` (4) and `thal` (2).
- 5 `eda_*.png` files exist in `reports/figures/`.
- **Write down** (for the model card): % male, age range/mean, % positive.

---

## PHASE 4 — Preprocessing pipeline (≈10 min)

Design: numeric → median impute (+ scale for linear model only); categorical → most-frequent impute + one-hot (`handle_unknown="ignore"`). Everything lives **inside** the sklearn `Pipeline`, so there is **no data leakage** in cross-validation.

### 4.1 Create `src/features.py`

```python
"""Preprocessing: imputation, one-hot encoding, optional scaling."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """scale=True for linear models, False for tree models (scaling is unnecessary)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), config.NUMERIC),
            ("cat", Pipeline(cat_steps), config.CATEGORICAL),
        ]
    )
```

### DONE WHEN
```bash
python -c "
from src.data import load_data, coerce_features
from src.features import build_preprocessor
d = load_data(); X = coerce_features(d)
Z = build_preprocessor(True).fit_transform(X)
print(Z.shape)
"
```
prints a shape starting with `303` and no errors (second number ≈ 28).

---

## PHASE 5 — Model training, comparison, evaluation (≈25 min)

### 5.1 Create `src/train.py`

```python
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
```

### 5.2 Run
```bash
python -m src.train
```

### DONE WHEN
- No traceback. `models/model.joblib`, `models/metrics.json`, `reports/model_comparison.csv`, `data/processed/test.csv`, `reports/figures/roc_curve.png`, `reports/figures/confusion_matrix.png` all exist.
- Dummy model CV ROC-AUC ≈ `0.5`; real models ≈ `0.85–0.92`. **If a real model is below 0.75 → something is wrong; see Troubleshooting T5.**
- A line `Selected model: random_forest` or `gradient_boosting` is printed.

---

## PHASE 6 — Explainability: feature importance + SHAP (≈20 min)

### 6.1 Create `src/explain.py`

```python
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
```

### 6.2 Run
```bash
python -m src.explain
```

### DONE WHEN
- `reports/figures/` contains `permutation_importance.png`, `shap_summary.png`, `shap_bar.png`.
- Open `shap_bar.png`: typical top features are `thal`, `cp`, `ca`, `oldpeak`, `thalach`, `exang` (one-hot columns appear separately). If all bars are ≈0 → Troubleshooting T6.

---

## PHASE 7 — Inference module (≈10 min)

### 7.1 Create `src/predict.py`

```python
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
    """If model file is missing (e.g. fresh cloud checkout), train it (~20 s)."""
    if not config.MODEL_PATH.exists():
        from . import explain, train
        train.main()
        explain.main()


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
```

### 7.2 Quick check
```bash
python -c "
from src.predict import load_bundle, predict_one, explain_one, risk_band
pipe, ex = load_bundle()
row = dict(age=67, sex=1, cp=4, trestbps=160, chol=286, fbs=0, restecg=2,
           thalach=108, exang=1, oldpeak=1.5, slope=2, ca=3, thal=3)
p = predict_one(pipe, row); print(round(p,3), risk_band(p)); print(explain_one(pipe, ex, row).head(5))
"
```

### DONE WHEN
- Probability printed is between 0 and 1 (this record is a real positive case; expect high, e.g. > 0.6).
- A 5-row series of feature contributions is printed.

---

## PHASE 8 — Minimal Streamlit app (≈25 min)

### 8.1 Create `app.py`

```python
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
```

### 8.2 Run locally
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

### DONE WHEN (manual click-test)
1. The yellow disclaimer banner is visible.
2. Click **"Load example: higher-risk profile"** → probability is high and the red bar chart changes. Click **"lower-risk profile"** → probability drops clearly.
3. Moving the age slider changes the probability/chart.
4. All four tabs render without red error boxes (the *Limitations* tab may say "MODEL_CARD.md not found" until Phase 9).
5. If an example does not give a clearly different band, that is fine — pick any other real rows from the dataset, but **report the true model output**; do not fake it.

---

## PHASE 9 — Documentation: model card, limitations, README (≈15 min)

### 9.1 Create `MODEL_CARD.md`
Replace every `<<...>>` with the real value from `reports/eda_summary.txt` or `models/metrics.json`.

```markdown
## Model card — CarePredict

**Intended use:** educational demonstration of an explainable ML pipeline. **Not** a medical device. **Not** a diagnosis. Not validated for clinical use.

**Data:** UCI Heart Disease, Cleveland subset (public, de-identified), <<303>> patients, 13 features. Outcome = angiographic heart disease present (`num > 0`). Positive rate ≈ <<46%>>.

**Model:** <<selected_model>> inside a scikit-learn pipeline (median/most-frequent imputation, one-hot encoding). Compared with a dummy baseline and logistic regression using stratified 5-fold CV; final numbers from a stratified 80/20 hold-out set (n_test = <<61>>).

**Results (hold-out):** ROC-AUC <<x.xxx>> (bootstrap 95% CI <<a–b>>), recall <<x.xxx>>, precision <<x.xxx>>. The CI is wide because the test set is tiny.

### Known limitations and bias
- **Small sample** (≈300 patients): estimates are noisy; small differences between models are not significant.
- **Single source & old data** (collected in the 1980s at one US clinic): may not match modern populations, equipment, or treatment practice.
- **Sex imbalance:** about <<68%>> of records are male; **no ethnicity/race information** is recorded, so fairness across groups cannot be assessed. Performance for women and under-represented groups is likely worse and is **not** measured.
- **Age range** is roughly <<29–77>>; the model should not be used outside it.
- **Label meaning:** the outcome is presence of angiographic disease at examination, not future events.
- **Selection bias:** patients were referred for cardiac tests, so they are not a general-population sample.
- **Missing values** (`ca`, `thal`) were imputed with the most frequent value, which can hide uncertainty.
- **Generalisation:** no external validation set was used. Expect lower performance on new hospitals.
- **Probabilities are not calibrated clinical risks;** "low/intermediate/high" bands are illustrative.
- **SHAP explains the model, not biology.** Feature contributions describe associations in this dataset, not causes.
- **Not a medical diagnosis.** Always consult a qualified clinician.
```

### 9.2 Create `README.md`
Must contain: title, one-paragraph description, the disclaimer sentence, live-demo link (fill after Phase 11), screenshot (optional), "How to run" (the 5 commands below), results table (copied from `reports/model_comparison.csv`), link to `MODEL_CARD.md`.

```bash
pip install -r requirements.txt
python -m src.eda
python -m src.train
python -m src.explain
streamlit run app.py
```

### DONE WHEN
- No `<<` placeholders remain: `grep -n "<<" MODEL_CARD.md README.md` prints nothing.
- App's "Limitations & bias" tab now shows the card.

---

## PHASE 10 — Tests & version pinning (≈10 min)

### 10.1 Create `tests/test_smoke.py`

```python
import math

from src.predict import explain_one, load_bundle, predict_one

ROW = dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2,
           thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6)


def test_probability_in_range():
    pipe, _ = load_bundle()
    p = predict_one(pipe, ROW)
    assert 0.0 <= p <= 1.0


def test_missing_value_is_handled():
    pipe, _ = load_bundle()
    row = dict(ROW, ca=float("nan"), thal=float("nan"))
    p = predict_one(pipe, row)
    assert not math.isnan(p)


def test_explanation_covers_all_features():
    pipe, ex = load_bundle()
    s = explain_one(pipe, ex, ROW)
    assert len(s) == 13
```

### 10.2 Run
```bash
python -m pytest -q
```
Expect `3 passed`.

### 10.3 Pin exact versions (prevents cloud/pickle mismatch)

```bash
pip freeze | grep -iE "^(numpy|pandas|scikit-learn|shap|matplotlib|streamlit|joblib)=="
```
(Windows PowerShell: `pip freeze | findstr /i "numpy pandas scikit-learn shap matplotlib streamlit joblib"`.)
**Replace the contents of `requirements.txt`** with exactly those lines (e.g. `scikit-learn==1.5.2`). Then **re-run `python -m src.train && python -m src.explain`** so the saved model matches the pinned versions.

### DONE WHEN
- `3 passed`.
- `requirements.txt` has only `==` pins.

---

## PHASE 11 — Free cloud deployment (≈20 min)

### Option A (primary): Streamlit Community Cloud — free

1. **Check the repo has everything:** `app.py`, `requirements.txt`, `src/`, `models/model.joblib`, `models/metrics.json`, `reports/`, `data/`, `MODEL_CARD.md`. (Files are small — fine for Git.)
2. **Create a public GitHub repo** named `carepredict` (no template), then:
   ```bash
   git init
   git add .
   git commit -m "CarePredict: healthcare risk prediction"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/carepredict.git
   git push -u origin main
   ```
3. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app** → *Deploy a public app from GitHub*.
4. Fill in: Repository = `<YOUR_USERNAME>/carepredict`, Branch = `main`, Main file path = `app.py`.
5. Click **Advanced settings** → **Python version** = `3.11` (or the same minor version you used locally).
6. Click **Deploy**. First build takes 3–6 minutes (installs shap/scikit-learn).
7. Copy the URL (`https://<something>.streamlit.app`) into `README.md`, commit and push.

**Safety net already built in:** if `models/model.joblib` is missing on the cloud, `ensure_model()` trains the model automatically at first load (~20–40 s).

### Option B (backup): Hugging Face Spaces — free
1. https://huggingface.co → New Space → SDK **Streamlit** (if unavailable, choose **Docker**/"Blank" and ask the user) → Public.
2. Upload the same files (or `git push` to the Space repo). The Space `README.md` must start with:
   ```yaml
   ---
   title: CarePredict
   emoji: 🩺
   sdk: streamlit
   app_file: app.py
   ---
   ```
3. It builds from `requirements.txt` automatically.

### DONE WHEN
- Opening the public URL in an **incognito window** shows the app, banner, and a working prediction with SHAP chart.
- The URL is written in `README.md`.

---

## PHASE 12 — Demo script (practice once, ≈5 min talk)

Before the demo: **open the cloud URL 10 minutes early** (free apps sleep when idle; the first load wakes them). Also keep `streamlit run app.py` ready locally as a fallback, and record a 2-minute screen video as a second fallback.

| Min | Show | Say |
|---|---|---|
| 0:00 | Banner + title | "Public, de-identified UCI data; this is an explainable ML demo, **not** a diagnosis." |
| 0:30 | *Model performance* tab | "Baseline vs tree models compared with stratified CV; test set used once. Dummy ≈ 0.5 AUC, real models ≈ 0.9. The test set is small, so I report a bootstrap CI." |
| 1:30 | *Global explanations* tab | "Permutation importance and SHAP agree that features like `thal`, `cp`, `ca`, `oldpeak` matter most." |
| 2:30 | Prediction tab → click **higher-risk example** | "Probability is high; the red bars show which features pushed it up." |
| 3:30 | Click **lower-risk example**, then change age/ST depression slider | "Bars and probability update live — individual explanation per patient." |
| 4:15 | *Limitations & bias* tab | "Small, old, one-hospital data, mostly male, no ethnicity info, no external validation — so it cannot be used clinically." |
| 4:45 | Repo / `MODEL_CARD.md` | "All code, metrics, and limits are documented." |

**Likely questions & honest answers**
- *Why not just pick the best model overall?* — I deploy the best **tree** model because SHAP's fast exact TreeExplainer needs one; logistic regression is kept as a baseline for comparison.
- *Is 0.9 AUC reliable?* — Only roughly; n_test ≈ 61, so the CI is wide, and there is no external validation.
- *Why are missing values imputed with the mode?* — Only 6 cells missing; simple and leak-free inside the pipeline; noted as a limitation.
- *Does SHAP show causes?* — No, it explains the model's behaviour.

---

## 13. FINAL ACCEPTANCE CHECKLIST (tick every box)

- [ ] `python -m src.eda`, `python -m src.train`, `python -m src.explain` run end-to-end with no error.
- [ ] `reports/model_comparison.csv` lists dummy, logistic regression, random forest, gradient boosting.
- [ ] `models/metrics.json` has `test_metrics` including `roc_auc_ci95`.
- [ ] `python -m pytest -q` → 3 passed.
- [ ] App runs locally; both example buttons work; SHAP bar chart updates.
- [ ] Disclaimer banner visible on the first screen; `MODEL_CARD.md` has no placeholders.
- [ ] Code pushed to a public GitHub repo; `requirements.txt` fully pinned.
- [ ] Public cloud URL opens in incognito and predicts correctly; URL is in `README.md`.
- [ ] Demo rehearsed with Section 12.

---

## 14. TROUBLESHOOTING

| # | Symptom | Fix |
|---|---|---|
| T1 | `RuntimeError: Could not download the dataset` | Open the URL in a browser, save the page as `data/raw/processed.cleveland.data` (plain text, 303 lines, comma-separated), re-run. |
| T2 | `ModuleNotFoundError: No module named 'src'` | You are not in the repo root, or ran `python src/xxx.py`. Use `python -m src.xxx` from the root; ensure `src/__init__.py` exists. |
| T3 | `TypeError: OneHotEncoder ... sparse_output` | scikit-learn too old. `pip install -U "scikit-learn>=1.4"`. |
| T4 | SHAP import/numba/numpy error | Use Python 3.11 in a fresh venv; `pip install -U shap numpy`. |
| T5 | Real model AUC < 0.75 | Check `target` creation (`num > 0`), check `na_values="?"`, check you did not drop columns or shuffle labels; re-run Phase 2 checks. |
| T6 | SHAP plots empty / shape error | Print `sv.shape`; it must be `(303, n_transformed_columns)`. `positive_class_shap` already handles list and 3-D outputs — make sure you copied it exactly. |
| T7 | Streamlit: `StreamlitAPIException` about session state / slider value | Do **not** pass `value=` or `index=` to widgets that use `key=`; defaults come from `DEFAULTS` in `st.session_state`. |
| T8 | Cloud app crashes with `InconsistentVersionWarning` / unpickle error | Versions differ between local and cloud. Pin exact versions (Phase 10.3), retrain, commit new `models/model.joblib`, push; or delete the model file from the repo so it auto-trains on the cloud. |
| T9 | Cloud build fails on dependency install | Set Python to 3.11 in Advanced settings; loosen the failing pin to `>=` and redeploy. |
| T10 | Cloud app is "asleep" | Click "Wake up"/reload; do this before demos. |
| T11 | `use_container_width` deprecation warning | Harmless; ignore. |

---

## 15. OPTIONAL EXTRAS (only after the checklist is 100% done)
- Add a calibration curve plot (`sklearn.calibration.calibration_curve`) to Phase 5.
- Add a second dataset (e.g., Pima Indians Diabetes) as a robustness check — **do not** mix it into the deployed model.
- Add a decision-threshold slider in the app showing recall/precision trade-off on the test set.
- Add a GitHub Actions workflow that runs `pytest` on each push.

**End of agent.md**