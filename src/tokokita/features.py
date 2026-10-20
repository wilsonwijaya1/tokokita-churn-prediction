"""Leakage-safe feature engineering and labelling.

Contract
--------
* ``compute_features(..., as_of)`` only reads transactions with ``order_date <= as_of``.
* ``make_labels(..., as_of, horizon)`` only reads transactions in ``(as_of, as_of + horizon]``.
* The two windows never overlap, so a feature can never "see" the outcome it is predicting.

A *snapshot* is one ``as_of`` date. The modelling table (the *panel*) stacks one row per
active customer per snapshot.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config as C

NUMERIC_FEATURES = [
    # RFM
    "recency_days", "frequency_30d", "frequency_90d", "monetary_30d", "monetary_90d",
    "avg_order_value", "monetary_trend",
    # behaviour
    "unique_cats_90d", "category_diversity", "discount_rate", "channel_loyalty", "weekend_ratio",
    # lifecycle / tempo
    "tenure_days", "orders_total", "mean_gap_days", "recency_ratio",
    # short-term momentum
    "orders_7d", "spend_7d", "ewm_frequency",
    # business
    "clv_estimate", "clv_decile",
]
ONEHOT_FEATURES = (
    [f"age_{a}" for a in C.AGE_GROUPS]
    + ["gender_M"]
    + [f"acq_{a}" for a in C.ACQ_CHANNELS]
)
FEATURE_COLUMNS = NUMERIC_FEATURES + ONEHOT_FEATURES
ID_COLUMNS = ["customer_id", "as_of"]
LABEL_COLUMN = "churn_30d"


def _window(tx: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    """Orders in (as_of - days, as_of]."""
    return tx[tx["order_date"] > as_of - pd.Timedelta(days=days)]


def compute_features(transactions: pd.DataFrame, customers: pd.DataFrame, as_of,
                     active_only: bool = True,
                     active_window: int = C.ACTIVE_WINDOW_DAYS) -> pd.DataFrame:
    """Feature table for one snapshot, using only transactions up to and including ``as_of``.

    ``active_only=True`` keeps customers with at least one order in the last ``active_window``
    days (the business definition of an active customer). The flag exists so the *leaky*
    pipeline of the original coursework can be reproduced for comparison (see ``leakage.py``).
    """
    as_of = pd.Timestamp(as_of)
    tx = transactions.loc[transactions["order_date"] <= as_of]
    cust = customers.loc[customers["signup_date"] <= as_of].set_index("customer_id")

    g_all = tx.groupby("customer_id")["order_date"]
    last, first, n_total = g_all.max(), g_all.min(), g_all.size()
    f = pd.DataFrame({"last_order": last, "first_order": first, "orders_total": n_total})
    f = f.join(cust, how="inner")        # only customers that exist at as_of and have >= 1 order

    f["recency_days"] = (as_of - f["last_order"]).dt.days
    f["tenure_days"] = (as_of - f["signup_date"]).dt.days

    t90, t30, t7 = _window(tx, as_of, 90), _window(tx, as_of, 30), _window(tx, as_of, 7)
    t30_60 = _window(tx, as_of, 60)
    t30_60 = t30_60[t30_60["order_date"] <= as_of - pd.Timedelta(days=30)]

    f["frequency_90d"] = t90.groupby("customer_id").size()
    f["frequency_30d"] = t30.groupby("customer_id").size()
    f["monetary_90d"] = t90.groupby("customer_id")["order_value"].sum()
    f["monetary_30d"] = t30.groupby("customer_id")["order_value"].sum()
    prev30 = t30_60.groupby("customer_id")["order_value"].sum()
    f["orders_7d"] = t7.groupby("customer_id").size()
    f["spend_7d"] = t7.groupby("customer_id")["order_value"].sum()
    f["unique_cats_90d"] = t90.groupby("customer_id")["product_category"].nunique()
    f["discount_rate"] = t90.groupby("customer_id")["discount_used"].mean()
    f["weekend_ratio"] = t90.assign(w=t90["order_date"].dt.weekday >= 5).groupby("customer_id")["w"].mean()

    ch = t90.groupby(["customer_id", "channel"]).size().groupby(level=0).max()
    f["channel_loyalty"] = ch / f["frequency_90d"]

    age_days = (as_of - tx["order_date"]).dt.days.to_numpy()
    ewm = pd.Series(np.exp(-0.05 * age_days), index=tx["customer_id"].to_numpy())
    f["ewm_frequency"] = ewm.groupby(level=0).sum()          # half-life ~14 days

    for col in ["frequency_90d", "frequency_30d", "monetary_90d", "monetary_30d", "orders_7d", "spend_7d",
                "unique_cats_90d", "discount_rate", "weekend_ratio", "channel_loyalty", "ewm_frequency"]:
        f[col] = f[col].fillna(0.0)

    f["avg_order_value"] = (f["monetary_90d"] / f["frequency_90d"].replace(0, np.nan)).fillna(0.0)
    f["category_diversity"] = (f["unique_cats_90d"] / f["frequency_90d"].replace(0, np.nan)).fillna(0.0)
    prev30 = prev30.reindex(f.index).fillna(0.0)
    denom = (f["monetary_30d"] + prev30).replace(0, np.nan)
    f["monetary_trend"] = ((f["monetary_30d"] - prev30) / denom).fillna(0.0)   # in [-1, 1]

    span = (f["last_order"] - f["first_order"]).dt.days
    f["mean_gap_days"] = (span / (f["orders_total"] - 1).replace(0, np.nan)).fillna(float(active_window))
    f["recency_ratio"] = f["recency_days"] / f["mean_gap_days"].clip(lower=1.0)
    f["clv_estimate"] = f["avg_order_value"] * f["frequency_90d"] * 4

    if active_only:
        f = f[f["frequency_90d"] >= 1].copy()
    if len(f) >= 10:
        f["clv_decile"] = pd.qcut(f["clv_estimate"].rank(method="first"), 10, labels=False) + 1
    else:
        f["clv_decile"] = 1
    f["clv_decile"] = f["clv_decile"].astype(int)

    for a in C.AGE_GROUPS:
        f[f"age_{a}"] = (f["age_group"] == a).astype(int)
    f["gender_M"] = (f["gender"] == "M").astype(int)
    for a in C.ACQ_CHANNELS:
        f[f"acq_{a}"] = (f["acquisition_channel"] == a).astype(int)

    out = f[FEATURE_COLUMNS].astype(float).copy()
    out.insert(0, "as_of", as_of)
    out.insert(0, "customer_id", out.index.to_numpy())
    return out.reset_index(drop=True)


def make_labels(transactions: pd.DataFrame, customer_ids: Iterable[str], as_of,
                horizon: int = C.HORIZON_DAYS) -> pd.Series:
    """churn_30d = 1 if the customer places no order in (as_of, as_of + horizon]."""
    as_of = pd.Timestamp(as_of)
    end = as_of + pd.Timedelta(days=horizon)
    w = transactions.loc[(transactions["order_date"] > as_of) & (transactions["order_date"] <= end), "customer_id"]
    bought = set(w.unique())
    ids = pd.Index(list(customer_ids))
    return pd.Series((~ids.isin(bought)).astype(int), index=ids, name=LABEL_COLUMN)


def snapshot_dates(first: str = C.FIRST_SNAPSHOT, last: str = C.LAST_SNAPSHOT) -> list[pd.Timestamp]:
    """Snapshots on the 1st and 15th of every month."""
    months = pd.date_range(first, last, freq="MS")
    dates = sorted(list(months) + [m + pd.Timedelta(days=14) for m in months])
    return [d for d in dates if d <= pd.Timestamp(last)]


def build_panel(transactions: pd.DataFrame, customers: pd.DataFrame, as_ofs: Iterable,
                horizon: int = C.HORIZON_DAYS, data_end: str = C.DATA_END) -> pd.DataFrame:
    """Stack leakage-safe features and labels for every snapshot in ``as_ofs``."""
    parts = []
    for as_of in as_ofs:
        as_of = pd.Timestamp(as_of)
        if as_of + pd.Timedelta(days=horizon) > pd.Timestamp(data_end):
            raise ValueError(f"Label window for {as_of.date()} extends beyond the data end ({data_end}).")
        feats = compute_features(transactions, customers, as_of)
        labels = make_labels(transactions, feats["customer_id"], as_of, horizon)
        feats[LABEL_COLUMN] = labels.to_numpy()
        parts.append(feats)
    return pd.concat(parts, ignore_index=True)
