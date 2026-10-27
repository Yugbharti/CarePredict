# CarePredict — Healthcare Risk Prediction & Analytics Dashboard

CarePredict estimates the probability that a patient has heart disease from 13 clinical features, explains each prediction with SHAP, and provides an interactive 7-tab analytics dashboard along with a full auto-generated results report.

> **Disclaimer:** **Educational / research demo only — NOT a medical diagnosis.** Built on the public, de-identified UCI Heart Disease (Cleveland) dataset (303 patients). Do not use it to make health decisions. Consult a qualified clinician.

## How to Run

```bash
pip install -r requirements.txt
python -m src.eda
python -m src.train
python -m src.explain
python -m src.analysis
python -m src.report
streamlit run app.py
```

## Results Summary

### 1. Model Comparison (Stratified 5-Fold Cross Validation + OOF)

| Model | CV ROC-AUC (mean ± SD) | Recall | Precision | F1 | Accuracy | OOF AUC (all 303) |
|---|---|---|---|---|---|---|
| `dummy_most_frequent` | 0.500 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.541 | 0.500 |
| `logistic_regression` | 0.902 ± 0.015 | 0.810 | 0.873 | 0.837 | 0.855 | 0.915 |
| `random_forest` (Selected) | 0.897 ± 0.030 | 0.774 | 0.788 | 0.776 | 0.797 | 0.916 |
| `gradient_boosting` | 0.869 ± 0.027 | 0.774 | 0.776 | 0.770 | 0.789 | 0.895 |

### 2. Final Model Performance (`random_forest`)

| Evaluation | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **Hold-out test set** (n=61) | **0.949** (95% CI: 0.886–0.991) | 0.964 | 0.818 | 0.885 | 0.885 |
| **Out-of-fold (OOF)**, all 303 records (t=0.5) | **0.916** (95% CI: 0.883–0.948) | 0.799 | 0.835 | 0.816 | 0.835 |

- **Brier score (OOF):** 0.12
- **Average precision (OOF):** 0.91

### 3. Key Visualizations

| Cross-Validation Model Comparison | Out-of-Fold ROC Curves | Precision-Recall (OOF) |
|---|---|---|
| ![Model Comparison](reports/figures/model_comparison_cv.png) | ![OOF ROC](reports/figures/roc_overlay_oof.png) | ![PR Curve](reports/figures/pr_curve_oof.png) |

## Dashboard Features

The Streamlit dashboard (`app.py`) provides 7 interactive analytical tabs:
1. **📊 Overview:** 6 high-level KPIs, full ML architecture diagram, cross-validated model benchmark chart, and summary findings.
2. **🔎 Data Explorer:** Interactive filters by sex and age range, distributions split by target class, categorical rates, Spearman correlation matrix, and missing value profiling.
3. **🧪 Model Lab:** Multi-model ROC overlay, PR curve, reliability calibration plot, score separation histogram, and interactive threshold slider with real-time 2×2 confusion matrix.
4. **🧠 Explainability:** Global mean |SHAP| feature ranking, SHAP beeswarm summary plot, interactive single-feature SHAP dependence plots, and permutation importance.
5. **🎛️ Patient Simulator:** Real-time patient risk calculator with preset low/high risk buttons, patient percentile indicator, individual waterfall/bar SHAP attribution, and 1D what-if sensitivity curves.
6. **🧬 Subgroups & Errors:** Granular breakdown across demographic subgroups (sex, age bands, chest pain types) and interactive diagnostic view of misclassified cases (false positives/false negatives).
7. **📄 Full Report:** In-app rendering and one-click markdown download of the complete 11-section research report ([`reports/REPORT.md`](reports/REPORT.md)).

## Documentation
- Full results and methodological report: [reports/REPORT.md](reports/REPORT.md)
- Model card and limitations: [MODEL_CARD.md](MODEL_CARD.md)
