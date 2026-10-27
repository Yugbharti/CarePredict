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
