"""Synthetic TokoKita data generator.

The generator is a small, explicit *customer lifecycle* simulation:

* Every customer has a latent value tier, an individual purchase rate, and a latent
  defection date drawn from an exponential lifetime (the hazard depends on tier,
  acquisition channel, age group and discount propensity).
* In the weeks before defection the purchase rate winds down, which makes churn partly
  predictable from behaviour (falling frequency, growing recency) rather than pure noise.
* Orders are Poisson draws from a day-level rate, with weekday and Nov/Dec seasonality.

Latent columns (tier, defection date, ...) are returned in a separate ``latent`` table
that the feature pipeline never touches; they exist only for sanity checks and to document
the data-generating process.

Everything is driven by one ``numpy.random.Generator`` seeded from ``seed``, so the output
is bit-for-bit reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# Data-generating-process parameters (documented in the README).
TIER_PROBS = [0.50, 0.35, 0.15]
TIER_MONTHLY_ORDERS = {"low": 0.9, "medium": 2.0, "high": 4.0}
TIER_ORDER_VALUE = {"low": 150_000, "medium": 350_000, "high": 800_000}   # IDR
TIER_MONTHLY_HAZARD = {"low": 0.10, "medium": 0.06, "high": 0.04}
ACQ_HAZARD_MULT = {"organic": 1.0, "paid_social": 1.3, "paid_search": 1.1, "referral": 0.7, "email": 0.9}
AGE_HAZARD_MULT = {"18-25": 1.2, "26-35": 1.0, "36-45": 0.95, "46-55": 0.9, "55+": 0.95}
AGE_PROBS = [0.15, 0.35, 0.25, 0.15, 0.10]
ACQ_PROBS = [0.30, 0.25, 0.20, 0.15, 0.10]
CATEGORY_BASE_PROBS = np.array([0.20, 0.30, 0.15, 0.15, 0.10, 0.10])
CHANNEL_BASE_PROBS = np.array([0.40, 0.35, 0.25])
SEASON_MULT = {11: 1.15, 12: 1.20}      # 11.11 / 12.12 / year-end campaigns
WEEKEND_MULT = 1.15


@dataclass
class Dataset:
    customers: pd.DataFrame      # what a CRM would know
    transactions: pd.DataFrame   # order log
    latent: pd.DataFrame         # simulation ground truth, NOT available to models


def _cum_choice(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Vectorised categorical draw: ``probs`` is (n, k); returns an int index per row."""
    cum = np.cumsum(probs, axis=1)
    u = rng.random(len(probs))[:, None]
    return np.minimum((u > cum).sum(axis=1), probs.shape[1] - 1)


def generate_dataset(n_customers: int = C.N_CUSTOMERS, seed: int = C.SEED,
                     start: str = C.DATA_START, signup_end: str = C.SIGNUP_END,
                     end: str = C.DATA_END) -> Dataset:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n_days = len(days)
    signup_span = (pd.Timestamp(signup_end) - pd.Timestamp(start)).days + 1

    customer_id = np.array([f"C{i:05d}" for i in range(n_customers)])
    signup_idx = rng.integers(0, signup_span, n_customers)
    age = rng.choice(C.AGE_GROUPS, n_customers, p=AGE_PROBS)
    gender = rng.choice(C.GENDERS, n_customers, p=[0.48, 0.52])
    acq = rng.choice(C.ACQ_CHANNELS, n_customers, p=ACQ_PROBS)

    # --- latent customer traits ---------------------------------------------
    tier = rng.choice(C.TIERS, n_customers, p=TIER_PROBS)
    monthly_orders = np.array([TIER_MONTHLY_ORDERS[t] for t in tier]) * rng.gamma(4.0, 0.25, n_customers)
    disc_prop = rng.beta(2.0, 6.0, n_customers)                      # mean 0.25
    hazard = (np.array([TIER_MONTHLY_HAZARD[t] for t in tier])
              * np.array([ACQ_HAZARD_MULT[a] for a in acq])
              * np.array([AGE_HAZARD_MULT[a] for a in age])
              * (0.6 + 1.6 * disc_prop))                              # mean multiplier ~1
    lifetime_days = rng.exponential(30.0 / hazard)
    defect_idx = signup_idx + np.round(lifetime_days).astype(int)
    wind_down = rng.integers(20, 61, n_customers)
    value_scale = rng.lognormal(0.0, 0.25, n_customers)
    cat_pref = rng.dirichlet(CATEGORY_BASE_PROBS * 8, n_customers)
    ch_pref = rng.dirichlet(CHANNEL_BASE_PROBS * 4, n_customers)

    # --- day-level order intensity ------------------------------------------
    day_idx = np.arange(n_days)[None, :]
    months = days.month.to_numpy()
    season = np.array([SEASON_MULT.get(int(m), 1.0) for m in months])
    weekday = np.where(days.weekday.to_numpy() >= 5, WEEKEND_MULT, 1.0)
    calendar_mult = (season * weekday)[None, :]

    ttd = defect_idx[:, None] - day_idx                               # days until defection
    activity = np.where(ttd <= 0, 0.0,
                        np.where(ttd < wind_down[:, None], 0.15 + 0.85 * ttd / wind_down[:, None], 1.0))
    started = (day_idx >= signup_idx[:, None]).astype(float)
    lam = (monthly_orders / 30.0)[:, None] * calendar_mult * activity * started
    counts = rng.poisson(lam)

    cust_pos, day_pos = np.nonzero(counts)
    reps = counts[cust_pos, day_pos]
    cust_pos = np.repeat(cust_pos, reps)
    day_pos = np.repeat(day_pos, reps)
    n_orders = len(cust_pos)

    tier_value = np.array([TIER_ORDER_VALUE[t] for t in tier])[cust_pos]
    value = tier_value * value_scale[cust_pos] * rng.lognormal(-0.5 * 0.35 ** 2, 0.35, n_orders)
    value = np.maximum(50_000, np.round(value, -2))
    category = np.array(C.CATEGORIES)[_cum_choice(rng, cat_pref[cust_pos])]
    channel = np.array(C.ORDER_CHANNELS)[_cum_choice(rng, ch_pref[cust_pos])]
    discount = rng.random(n_orders) < disc_prop[cust_pos]

    transactions = pd.DataFrame({
        "customer_id": customer_id[cust_pos],
        "order_date": days[day_pos],
        "order_value": value,
        "product_category": category,
        "channel": channel,
        "discount_used": discount,
    }).sort_values(["customer_id", "order_date"], kind="stable").reset_index(drop=True)
    transactions.insert(0, "order_id", [f"O{i:07d}" for i in range(len(transactions))])

    customers = pd.DataFrame({
        "customer_id": customer_id,
        "signup_date": days[signup_idx],
        "age_group": age,
        "gender": gender,
        "acquisition_channel": acq,
    })
    latent = pd.DataFrame({
        "customer_id": customer_id,
        "value_tier": tier,
        "monthly_orders": monthly_orders,
        "discount_propensity": disc_prop,
        "defection_date": [days[i] if i < n_days else pd.NaT for i in defect_idx],
        "wind_down_days": wind_down,
    })
    return Dataset(customers=customers, transactions=transactions, latent=latent)


def save_dataset(ds: Dataset, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ds.customers.to_csv(directory / "customers.csv", index=False)
    ds.transactions.to_csv(directory / "transactions.csv", index=False)
    ds.latent.to_csv(directory / "latent_ground_truth.csv", index=False)
