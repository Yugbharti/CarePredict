# CarePredict — Heart-Disease Risk Estimator

CarePredict estimates the probability that a patient has heart disease from 13 clinical features, explains each prediction with SHAP, and clearly states it is NOT a medical diagnosis.

> **Disclaimer:** **Educational demo only — NOT a medical diagnosis.** Built on a small, old, single-hospital public dataset (UCI Cleveland, 303 patients). Do not use it to make health decisions. Consult a qualified clinician.

## Live Demo
- **Public URL:** *Pending cloud deployment (see Phase 11)*

## How to Run

```bash
pip install -r requirements.txt
python -m src.eda
python -m src.train
python -m src.explain
streamlit run app.py
```

## Model Comparison (Stratified 5-Fold Cross Validation)

Results from training cross-validation (`reports/model_comparison.csv`):

| Model | Deployable Tree Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `dummy_most_frequent` | False | 0.500 ± 0.000 | 0.541 ± 0.006 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| `logistic_regression` | False | 0.902 ± 0.015 | 0.855 ± 0.024 | 0.873 ± 0.062 | 0.810 ± 0.062 | 0.837 ± 0.026 |
| `random_forest` | True | 0.897 ± 0.030 | 0.797 ± 0.026 | 0.788 ± 0.053 | 0.774 ± 0.087 | 0.776 ± 0.035 |
| `gradient_boosting` | True | 0.869 ± 0.027 | 0.789 ± 0.037 | 0.776 ± 0.065 | 0.774 ± 0.083 | 0.770 ± 0.043 |

### Hold-out Test Set Performance (`random_forest`)
- **ROC-AUC:** 0.949 (Bootstrap 95% CI: 0.886–0.991)
- **Accuracy:** 0.885
- **Precision:** 0.818
- **Recall:** 0.964
- **F1 Score:** 0.885

## Model Card & Limitations
For detailed documentation on dataset characteristics, potential bias, fairness, and clinical limitations, see [MODEL_CARD.md](MODEL_CARD.md).
