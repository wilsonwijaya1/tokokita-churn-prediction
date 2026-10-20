"""Model zoo and probability calibration.

Differences from the original coursework:

* Logistic Regression is trained on *standardised* features (it was unscaled and failed to converge).
* No ``class_weight`` / ``scale_pos_weight`` / ``is_unbalance``: churn is ~1/3 of rows, which is not
  severe, and re-weighting classes systematically distorts probabilities (the original ECE was ~0.2).
  We rank with the raw model and then calibrate with isotonic regression fitted on the validation set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C
from .features import FEATURE_COLUMNS, LABEL_COLUMN

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


def build_models(seed: int = C.SEED) -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=25, n_jobs=4, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss",
            n_jobs=4, random_state=seed, tree_method="hist"),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=3, num_leaves=8, learning_rate=0.04, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=40, reg_lambda=5.0, n_jobs=4, random_state=seed, verbose=-1),
    }


def fit_models(train: pd.DataFrame, seed: int = C.SEED, feature_cols: list[str] | None = None) -> dict[str, object]:
    feature_cols = feature_cols or FEATURE_COLUMNS
    models = build_models(seed)
    for m in models.values():
        m.fit(train[feature_cols].to_numpy(), train[LABEL_COLUMN].to_numpy())
    return models


def predict_proba(models: dict[str, object], df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict[str, np.ndarray]:
    feature_cols = feature_cols or FEATURE_COLUMNS
    X = df[feature_cols].to_numpy()
    return {k: m.predict_proba(X)[:, 1] for k, m in models.items()}


def fit_isotonic(raw_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    return iso
