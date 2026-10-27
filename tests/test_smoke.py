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
