"""Data: UCI Default of Credit Card Clients, a synthetic look-alike, the credit
default models and drift injection for monitoring slices."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "default"
FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def download_uci(dest: str | Path = "data/uci_default.csv") -> Path:
    """Download the UCI dataset and save it as CSV with a ``default`` target."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    xls_name = next(n for n in outer.namelist() if n.endswith(".xls"))
    df = pd.read_excel(io.BytesIO(outer.read(xls_name)), header=1, engine="xlrd")
    df = df.rename(columns={"default payment next month": TARGET}).drop(columns=["ID"])
    df[FEATURES + [TARGET]].to_csv(dest, index=False)
    return dest


def load_uci(path: str | Path = "data/uci_default.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        download_uci(path)
    return pd.read_csv(path)


def synthetic_credit(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic credit-card portfolio with UCI-like columns and a logistic
    default mechanism (about 22% defaults). Used for offline tests and the
    bundled sample; it is not real customer data."""
    rng = np.random.default_rng(seed)
    limit = np.round(np.exp(rng.normal(11.8, 0.75, n)), -4).clip(10_000, 1_000_000)
    sex = rng.choice([1, 2], n, p=[0.4, 0.6])
    edu = rng.choice([1, 2, 3, 4, 5, 6, 0], n, p=[0.35, 0.47, 0.16, 0.005, 0.01, 0.003, 0.002])
    mar = rng.choice([1, 2, 3, 0], n, p=[0.455, 0.532, 0.011, 0.002])
    age = rng.gamma(9, 4, n).round().clip(21, 79)
    risk = rng.normal(0, 1, n) - 0.35 * (np.log(limit) - 11.8)
    pay0 = np.clip(np.round(risk * 1.2 - 0.3 + rng.normal(0, 0.6, n)), -2, 8)
    pay2 = np.clip(np.round(0.7 * pay0 + rng.normal(0, 0.7, n)), -2, 8)
    util = 1 / (1 + np.exp(-(risk + rng.normal(0, 1, n))))
    bill1 = np.round(limit * util * rng.uniform(0.2, 1.0, n), -1)
    pay_amt1 = np.round(bill1 * rng.beta(1, 6, n) + rng.exponential(1500, n), -1)
    logit = -1.5 + 0.75 * pay0 + 0.2 * pay2 - 0.25 * (np.log(limit) - 11.8) + 0.004 * (age - 35) - 0.1 * (sex == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "LIMIT_BAL": limit, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar, "AGE": age,
        "PAY_0": pay0, "PAY_2": pay2, "BILL_AMT1": bill1, "PAY_AMT1": pay_amt1, TARGET: y,
    })


def train_models(train: pd.DataFrame, features: list[str] | None = None, seed: int = 0) -> dict:
    """Fit the logistic-regression and gradient-boosting PD models."""
    features = features or [c for c in train.columns if c != TARGET]
    X, y = train[features], train[TARGET]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                        l2_regularization=1.0, random_state=seed)
    return {"logit": lr.fit(X, y), "gbm": gb.fit(X, y), "features": features}


def score(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Return df with ``pd_logit`` and ``pd_gbm`` score columns."""
    out = df.copy()
    X = df[models["features"]]
    out["pd_logit"] = models["logit"].predict_proba(X)[:, 1]
    out["pd_gbm"] = models["gbm"].predict_proba(X)[:, 1]
    return out


# ---------------------------------------------------------------- drift injection

def covariate_drift(pool: pd.DataFrame, n: int, delta: float, rng, column: str = "LIMIT_BAL") -> pd.DataFrame:
    """Resample with weights exp(-delta * z), z the standardized log of ``column``.
    Shifts the portfolio toward lower limits; outcomes follow the rows."""
    x = np.log(pool[column].to_numpy(dtype=float).clip(1))
    z = (x - x.mean()) / x.std()
    w = np.exp(-delta * z)
    idx = rng.choice(len(pool), n, replace=True, p=w / w.sum())
    return pool.iloc[idx].reset_index(drop=True)


def label_drift(pool: pd.DataFrame, n: int, delta: float, rng) -> pd.DataFrame:
    """Prior shift: resample so the default rate is multiplied by (1 + delta)
    while P(features | outcome) is unchanged."""
    y = pool[TARGET].to_numpy()
    base = y.mean()
    target = min(base * (1 + delta), 0.99)
    w = np.where(y == 1, target / base, (1 - target) / (1 - base))
    idx = rng.choice(len(pool), n, replace=True, p=w / w.sum())
    return pool.iloc[idx].reset_index(drop=True)


def calibration_drift(pool: pd.DataFrame, n: int, delta: float, rng) -> pd.DataFrame:
    """Concept drift: draw rows as in a stable period, then turn a fraction
    ``delta`` of non-defaults into defaults at random. Features and scores keep
    their distribution, but the PD is too low and discrimination weakens."""
    idx = rng.integers(0, len(pool), n)
    out = pool.iloc[idx].reset_index(drop=True)
    y = out[TARGET].to_numpy().copy()
    flip = (y == 0) & (rng.random(n) < delta)
    y[flip] = 1
    out[TARGET] = y
    return out


DRIFTS = {"covariate": covariate_drift, "label": label_drift, "calibration": calibration_drift}
