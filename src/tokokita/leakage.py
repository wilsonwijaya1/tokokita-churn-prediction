"""Leakage tooling: reproduce the failure mode the original analysis uncovered, and audit for it.

In the original coursework, the first evaluation (section B.6) scored features computed at
``2024-12-01`` against a label describing November (``2024-11-01 .. 2024-12-01``). Those features
already contained the outcome: a customer with no order in November necessarily had
``recency_days > 30`` on 1 December, so one feature separated the classes perfectly (AUC 1.0).
The original analysis noticed this itself and switched to an out-of-time evaluation (B.7).

This module keeps that insight as reusable guardrails: ``build_leaky_panel`` recreates the
overlapping-window mistake on the new data so it can be shown side by side with the clean pipeline,
and ``audit_single_feature_auc`` is an automated check that flags such features.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import FEATURE_COLUMNS, LABEL_COLUMN, compute_features, make_labels


def build_leaky_panel(transactions: pd.DataFrame, customers: pd.DataFrame, as_ofs: Iterable,
                      horizon: int = C.HORIZON_DAYS) -> pd.DataFrame:
    """Same customers and labels as the clean panel, but features are computed at the END of the
    label window (``as_of + horizon``) instead of at ``as_of``. DO NOT use for modelling."""
    parts = []
    for as_of in as_ofs:
        as_of = pd.Timestamp(as_of)
        clean = compute_features(transactions, customers, as_of)
        leaky = compute_features(transactions, customers, as_of + pd.Timedelta(days=horizon), active_only=False)
        leaky = leaky.set_index("customer_id").reindex(clean["customer_id"])
        # customers with no order before the window end cannot be featurised; they cannot exist here
        leaky = leaky.reset_index()
        leaky["as_of"] = as_of
        leaky[LABEL_COLUMN] = make_labels(transactions, leaky["customer_id"], as_of, horizon).to_numpy()
        parts.append(leaky)
    return pd.concat(parts, ignore_index=True)


def audit_single_feature_auc(df: pd.DataFrame, feature_cols: list[str] | None = None,
                             label: str = LABEL_COLUMN) -> pd.DataFrame:
    """Direction-agnostic AUC of every feature on its own, sorted descending.

    A single honest feature almost never reaches 0.9 on a behavioural churn task; values near 1.0
    mean the feature encodes the label.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    rows = []
    for c in feature_cols:
        x = df[c].fillna(0.0)
        if x.nunique() < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(df[label], x)
        rows.append((c, auc, max(auc, 1 - auc)))
    out = pd.DataFrame(rows, columns=["feature", "auc", "strength"]).sort_values("strength", ascending=False)
    return out.reset_index(drop=True)


def flag_leaky_features(df: pd.DataFrame, feature_cols: list[str] | None = None, threshold: float = 0.90) -> list[str]:
    audit = audit_single_feature_auc(df, feature_cols)
    return audit.loc[audit["strength"] >= threshold, "feature"].tolist()
