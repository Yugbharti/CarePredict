## Model card — CarePredict

**Intended use:** educational demonstration of an explainable ML pipeline. **Not** a medical device. **Not** a diagnosis. Not validated for clinical use.

**Data:** UCI Heart Disease, Cleveland subset (public, de-identified), 303 patients, 13 features. Outcome = angiographic heart disease present (`num > 0`). Positive rate ≈ 46%.

**Model:** random_forest inside a scikit-learn pipeline (median/most-frequent imputation, one-hot encoding). Compared with a dummy baseline and logistic regression using stratified 5-fold CV; final numbers from a stratified 80/20 hold-out set (n_test = 61).

**Results (hold-out):** ROC-AUC 0.949 (bootstrap 95% CI 0.886–0.991), recall 0.964, precision 0.818. The CI is wide because the test set is tiny.

### Known limitations and bias
- **Small sample** (≈300 patients): estimates are noisy; small differences between models are not significant.
- **Single source & old data** (collected in the 1980s at one US clinic): may not match modern populations, equipment, or treatment practice.
- **Sex imbalance:** about 68% of records are male; **no ethnicity/race information** is recorded, so fairness across groups cannot be assessed. Performance for women and under-represented groups is likely worse and is **not** measured.
- **Age range** is roughly 29–77; the model should not be used outside it.
- **Label meaning:** the outcome is presence of angiographic disease at examination, not future events.
- **Selection bias:** patients were referred for cardiac tests, so they are not a general-population sample.
- **Missing values** (`ca`, `thal`) were imputed with the most frequent value, which can hide uncertainty.
- **Generalisation:** no external validation set was used. Expect lower performance on new hospitals.
- **Probabilities are not calibrated clinical risks;** "low/intermediate/high" bands are illustrative.
- **SHAP explains the model, not biology.** Feature contributions describe associations in this dataset, not causes.
- **Not a medical diagnosis.** Always consult a qualified clinician.
