import json

from src import config
from src.predict import ensure_model


def test_artifacts_exist_and_consistent():
    ensure_model()
    for path in (config.ANALYSIS_PATH, config.OOF_CSV, config.SHAP_CSV,
                 config.ERRORS_CSV, config.REPORT_MD):
        assert path.exists(), path
    a = json.loads(config.ANALYSIS_PATH.read_text())
    assert 0.5 < a["oof"]["roc_auc"] <= 1.0
    lo, hi = a["oof"]["roc_auc_ci95"]
    assert lo <= a["oof"]["roc_auc"] <= hi
    e = a["errors"]
    assert e["true_positives"] + e["true_negatives"] + e["false_positives"] + e["false_negatives"] == a["n_records"]


def test_report_has_all_sections():
    ensure_model()
    text = config.REPORT_MD.read_text()
    assert text.count("\n## ") + text.startswith("## ") == 11
    assert "nan" not in text.lower().replace("finance", "")
