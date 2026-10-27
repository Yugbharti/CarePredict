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
