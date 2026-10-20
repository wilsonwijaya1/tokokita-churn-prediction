"""Retention economics: net benefit as a function of the contact threshold.

Decision: send a retention offer (email + voucher) to every customer whose calibrated churn
probability is at least ``t``.

    net_benefit(t) = sum over contacted customers of
                       [ churned_i * save_rate * retention_value_i  -  contact_cost ]

* ``churned_i`` is the realised outcome (1 if the customer really lapsed).
* ``save_rate`` is the share of would-be churners the offer actually keeps. The original coursework
  implicitly assumed 100%; here it is an explicit, sensitivity-tested assumption.
* ``retention_value_i`` is customer-specific: contribution margin x retained months x the customer's
  trailing monthly spend. A high spender is worth more to save than a low spender.

All monetary values are in Indonesian rupiah (IDR). The parameters are ASSUMPTIONS, not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetentionEconomics:
    contact_cost: float = 35_000     # IDR per contact (email + expected voucher cost)
    save_rate: float = 0.15          # share of would-be churners retained by the offer
    margin: float = 0.25             # contribution margin on retained revenue
    retained_months: float = 6.0     # months of purchasing preserved when a customer is saved

    def with_(self, **kw) -> "RetentionEconomics":
        return replace(self, **kw)


def retention_value(monetary_90d, econ: RetentionEconomics) -> np.ndarray:
    """Margin earned over ``retained_months`` if the customer keeps buying at the trailing monthly rate."""
    return econ.margin * econ.retained_months * np.asarray(monetary_90d, dtype=float) / 3.0


def net_benefit(contacted, churned, value, econ: RetentionEconomics) -> float:
    contacted = np.asarray(contacted, dtype=bool)
    churned = np.asarray(churned, dtype=float)
    value = np.asarray(value, dtype=float)
    gain = (churned * econ.save_rate * value)[contacted].sum()
    return float(gain - contacted.sum() * econ.contact_cost)


def sweep_thresholds(prob, churned, value, econ: RetentionEconomics, thresholds=None,
                     n_snapshots: int = 1) -> pd.DataFrame:
    """Net benefit and contact statistics for every threshold in ``thresholds``."""
    prob, churned = np.asarray(prob, dtype=float), np.asarray(churned)
    thresholds = np.round(np.arange(0.0, 1.0001, 0.01), 2) if thresholds is None else thresholds
    n = len(prob)
    rows = []
    for t in thresholds:
        c = prob >= t
        nb = net_benefit(c, churned, value, econ)
        k = int(c.sum())
        rows.append({
            "threshold": float(t),
            "n_contacted": k,
            "share_contacted": k / n,
            "precision": float(churned[c].mean()) if k else float("nan"),
            "recall": float(churned[c].sum() / churned.sum()) if k else 0.0,
            "net_benefit": nb,
            "net_benefit_per_run": nb / n_snapshots,
            "net_benefit_per_1000_scored": nb / n * 1000,
            "roi": nb / (k * econ.contact_cost) if k else float("nan"),
        })
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame) -> pd.Series:
    """Row of the sweep with maximum net benefit (earliest threshold wins ties)."""
    return sweep.loc[sweep["net_benefit"].idxmax()]


def expected_value_policy(prob, value, econ: RetentionEconomics) -> np.ndarray:
    """Customer-specific rule: contact iff expected savings exceed the contact cost."""
    return np.asarray(prob) * econ.save_rate * np.asarray(value) > econ.contact_cost
