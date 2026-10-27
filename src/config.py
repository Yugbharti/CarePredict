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
