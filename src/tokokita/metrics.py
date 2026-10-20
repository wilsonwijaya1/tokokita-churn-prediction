"""Ranking and probability metrics, plus a customer-clustered bootstrap.

The same customer appears in several snapshots, so rows are not independent. The bootstrap
therefore resamples *customers* (all of their snapshots together), which gives honest, wider
intervals than a naive row bootstrap. The same resamples are reused for every model, so
differences between models are paired.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _top_k_index(score: np.ndarray, frac: float) -> np.ndarray:
    n = len(score)
    k = max(1, int(np.ceil(frac * n)))
    return np.argsort(-np.asarray(score), kind="stable")[:k]


def precision_at_k(y, score, frac: float = 0.10) -> float:
    """Share of true churners among the top ``frac`` highest-scored customers."""
    y = np.asarray(y)
    return float(y[_top_k_index(score, frac)].mean())


def recall_at_k(y, score, frac: float = 0.10) -> float:
    """Share of all churners captured in the top ``frac`` highest-scored customers."""
    y = np.asarray(y)
    total = y.sum()
    return float(y[_top_k_index(score, frac)].sum() / total) if total > 0 else float("nan")


def lift_at_k(y, score, frac: float = 0.10) -> float:
    return precision_at_k(y, score, frac) / float(np.mean(y))


def expected_calibration_error(y, prob, n_bins: int = 10) -> float:
    """Weighted ECE: sum_b (n_b / n) * |mean(y in b) - mean(p in b)| over equal-width probability bins.

    (The original coursework used the *unweighted* mean over bins, which over-penalises sparse bins.)
    """
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(ece)


def evaluate(y, prob) -> dict[str, float]:
    y, prob = np.asarray(y), np.asarray(prob)
    return {
        "AUC": float(roc_auc_score(y, prob)),
        "PR_AUC": float(average_precision_score(y, prob)),
        "Precision@10%": precision_at_k(y, prob, 0.10),
        "Recall@10%": recall_at_k(y, prob, 0.10),
        "Lift@10%": lift_at_k(y, prob, 0.10),
        "Brier": float(brier_score_loss(y, prob)),
        "ECE": expected_calibration_error(y, prob),
    }


def cluster_resamples(groups, n_boot: int = 1000, seed: int = 42) -> Iterator[np.ndarray]:
    """Yield row-index arrays for ``n_boot`` customer-level bootstrap resamples."""
    groups = np.asarray(groups)
    codes, uniques = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes, minlength=len(uniques))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniques), len(uniques))
        lens = sizes[pick]
        total = lens.sum()
        offsets = np.repeat(starts[pick] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        yield order[offsets + np.arange(total)]


def bootstrap_metrics(y, probs: dict[str, np.ndarray], groups, metric_fn: Callable = evaluate,
                      n_boot: int = 1000, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Per-model DataFrame of bootstrap replicates (rows = resamples, columns = metrics)."""
    y = np.asarray(y)
    probs = {k: np.asarray(v) for k, v in probs.items()}
    reps: dict[str, list] = {k: [] for k in probs}
    for idx in cluster_resamples(groups, n_boot, seed):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, p in probs.items():
            reps[k].append(metric_fn(yb, p[idx]))
    return {k: pd.DataFrame(v) for k, v in reps.items()}


def percentile_ci(replicates: pd.DataFrame | pd.Series, level: float = 0.95) -> pd.DataFrame:
    a = (1 - level) / 2
    q = replicates.quantile([a, 1 - a])
    return q.T.rename(columns={a: "lo", 1 - a: "hi"}) if isinstance(q, pd.DataFrame) else q
