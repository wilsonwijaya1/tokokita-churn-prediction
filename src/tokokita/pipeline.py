"""End-to-end experiment: data -> panel -> purged split -> models -> calibration -> CIs -> economics.

Both ``scripts/run_pipeline.py`` and the notebook call :func:`run_experiment`, so every number in
the README, the figures and the notebook comes from the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C
from .data import Dataset, generate_dataset
from .decision import (RetentionEconomics, best_threshold, expected_value_policy, net_benefit, retention_value,
                       sweep_thresholds)
from .features import FEATURE_COLUMNS, LABEL_COLUMN, build_panel, snapshot_dates
from .leakage import audit_single_feature_auc, build_leaky_panel
from .metrics import bootstrap_metrics, cluster_resamples, evaluate, percentile_ci
from .models import MODEL_NAMES, fit_isotonic, fit_models, predict_proba
from .splits import Split, assert_no_temporal_overlap, temporal_split

RANKING_METRICS = ["AUC", "PR_AUC", "Precision@10%", "Recall@10%", "Lift@10%"]
CAL_METRICS = ["Brier", "ECE"]


@dataclass
class Experiment:
    dataset: Dataset
    panel: pd.DataFrame
    split: Split
    models: dict
    raw: dict            # {"val": {model: p}, "test": {model: p}}
    calibrated: dict     # same shape, isotonic-calibrated (fit on val)
    calibrators: dict
    selected: str
    val_metrics: pd.DataFrame
    test_metrics: pd.DataFrame            # raw ranking metrics + calibrated Brier/ECE
    test_metrics_uncalibrated: pd.DataFrame
    test_ci: pd.DataFrame                 # long format: model, metric, point, lo, hi
    paired_auc: pd.DataFrame
    leakage: dict
    importance: pd.DataFrame
    econ: RetentionEconomics
    val_sweep: pd.DataFrame
    test_sweep: pd.DataFrame
    chosen: pd.Series                     # threshold chosen on validation
    policies: pd.DataFrame
    policy_ci: dict
    sensitivity: pd.DataFrame
    ev_by_value: pd.DataFrame
    break_even_save_rate: float
    break_even_save_rate_flat: float
    extras: dict = field(default_factory=dict)


def _metrics_table(y, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({k: evaluate(y, p) for k, p in probs.items()}).T.loc[MODEL_NAMES]


def run_experiment(n_customers: int = C.N_CUSTOMERS, seed: int = C.SEED, n_boot: int = 1000,
                   econ: RetentionEconomics | None = None, verbose: bool = False) -> Experiment:
    econ = econ or RetentionEconomics()
    log = print if verbose else (lambda *a, **k: None)

    ds = generate_dataset(n_customers, seed)
    snaps = snapshot_dates()
    panel = build_panel(ds.transactions, ds.customers, snaps)
    split = temporal_split(panel)
    assert_no_temporal_overlap(split)
    log("panel", panel.shape, "train/val/test", len(split.train), len(split.val), len(split.test))

    models = fit_models(split.train, seed)
    y_val, y_test = split.val[LABEL_COLUMN].to_numpy(), split.test[LABEL_COLUMN].to_numpy()
    raw = {"val": predict_proba(models, split.val), "test": predict_proba(models, split.test)}
    calibrators = {k: fit_isotonic(raw["val"][k], y_val) for k in models}
    calibrated = {part: {k: calibrators[k].predict(p) for k, p in raw[part].items()} for part in raw}

    val_metrics = _metrics_table(y_val, raw["val"])
    selected = val_metrics["AUC"].idxmax()
    test_raw = _metrics_table(y_test, raw["test"])
    test_cal = _metrics_table(y_test, calibrated["test"])
    test_metrics = test_raw.copy()
    test_metrics[CAL_METRICS] = test_cal[CAL_METRICS]

    # --- customer-clustered bootstrap (paired across models) ---------------------------------
    groups = split.test["customer_id"].to_numpy()
    boot_raw = bootstrap_metrics(y_test, raw["test"], groups, n_boot=n_boot, seed=seed)
    boot_cal = bootstrap_metrics(y_test, calibrated["test"], groups, n_boot=n_boot, seed=seed)
    ci_rows = []
    for m in MODEL_NAMES:
        for metric in RANKING_METRICS:
            lo, hi = boot_raw[m][metric].quantile([0.025, 0.975])
            ci_rows.append((m, metric, test_metrics.loc[m, metric], lo, hi))
        for metric in CAL_METRICS:
            lo, hi = boot_cal[m][metric].quantile([0.025, 0.975])
            ci_rows.append((m, metric, test_metrics.loc[m, metric], lo, hi))
    test_ci = pd.DataFrame(ci_rows, columns=["model", "metric", "point", "lo", "hi"])

    pair_rows = []
    for m in MODEL_NAMES:
        d = boot_raw[selected]["AUC"].to_numpy() - boot_raw[m]["AUC"].to_numpy()
        pair_rows.append((m, test_metrics.loc[selected, "AUC"] - test_metrics.loc[m, "AUC"],
                          np.quantile(d, 0.025), np.quantile(d, 0.975)))
    paired_auc = pd.DataFrame(pair_rows, columns=["model", f"AUC({selected}) - AUC(model)", "lo", "hi"]).set_index("model")

    # --- leakage demonstration ----------------------------------------------------------------
    eval_snaps = sorted(set(split.test["as_of"]))
    leaky_test = build_leaky_panel(ds.transactions, ds.customers, eval_snaps)
    leaky_probs = predict_proba(models, leaky_test)        # models trained on CLEAN data, evaluated on leaky features
    y_leaky = leaky_test[LABEL_COLUMN].to_numpy()
    leaky_table = _metrics_table(y_leaky, leaky_probs)
    leakage = {
        "leaky_test": leaky_test,
        "leaky_metrics": leaky_table,
        "audit_leaky": audit_single_feature_auc(leaky_test),
        "audit_clean": audit_single_feature_auc(split.test),
        "recency_auc_leaky": float(audit_single_feature_auc(leaky_test, ["recency_days"])["auc"].iloc[0]),
        "recency_auc_clean": float(audit_single_feature_auc(split.test, ["recency_days"])["auc"].iloc[0]),
    }

    # --- permutation importance on validation (selected model) ---------------------------------
    pi = permutation_importance(models[selected], split.val[FEATURE_COLUMNS].to_numpy(), y_val,
                                scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=1)
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "auc_drop": pi.importances_mean, "std": pi.importances_std})
    importance = importance.sort_values("auc_drop", ascending=False).reset_index(drop=True)

    # --- retention economics --------------------------------------------------------------------
    val_value = retention_value(split.val["monetary_90d"], econ)
    test_value = retention_value(split.test["monetary_90d"], econ)
    n_test_snaps = split.test["as_of"].nunique()
    p_val, p_test = calibrated["val"][selected], calibrated["test"][selected]
    val_sweep = sweep_thresholds(p_val, y_val, val_value, econ, n_snapshots=split.val["as_of"].nunique())
    test_sweep = sweep_thresholds(p_test, y_test, test_value, econ, n_snapshots=n_test_snaps)
    chosen = best_threshold(val_sweep)
    t_star = float(chosen["threshold"])

    def _policy_row(name, mask):
        mask = np.asarray(mask, dtype=bool)
        nb = net_benefit(mask, y_test, test_value, econ)
        k = int(mask.sum())
        return {"policy": name, "share_contacted": k / len(mask), "precision": float(y_test[mask].mean()) if k else np.nan,
                "net_benefit_per_run": nb / n_test_snaps, "net_benefit_per_1000_scored": nb / len(mask) * 1000}

    top10 = np.zeros(len(y_test), bool)
    top10[np.argsort(-p_test, kind="stable")[: int(np.ceil(0.1 * len(y_test)))]] = True
    policies = pd.DataFrame([
        _policy_row("Contact nobody", np.zeros(len(y_test), bool)),
        _policy_row("Contact everyone", np.ones(len(y_test), bool)),
        _policy_row("Top 10% by risk", top10),
        _policy_row(f"Threshold p >= {t_star:.2f} (chosen on validation)", p_test >= t_star),
        _policy_row("Expected-value rule (per-customer)", expected_value_policy(p_test, test_value, econ)),
    ])

    # bootstrap CI of net benefit for the two policies (customer-clustered, same resamples as above)
    ev_mask = expected_value_policy(p_test, test_value, econ)
    flat_mask = p_test >= t_star
    boot = {"flat": [], "ev": [], "flat_prec": [], "ev_prec": []}
    for idx in cluster_resamples(groups, n_boot, seed):
        for key, mask in (("flat", flat_mask), ("ev", ev_mask)):
            c = mask[idx]
            boot[key].append(net_benefit(c, y_test[idx], test_value[idx], econ) / len(idx) * 1000)
            boot[key + "_prec"].append(y_test[idx][c].mean() if c.any() else np.nan)
    policy_ci = {}
    for key in ("flat", "ev"):
        arr = np.array(boot[key])
        policy_ci[key] = {"nb_per_1000_lo": float(np.quantile(arr, 0.025)), "nb_per_1000_hi": float(np.quantile(arr, 0.975)),
                          "prob_positive": float(np.mean(arr > 0)),
                          "precision_lo": float(np.nanquantile(boot[key + "_prec"], 0.025)),
                          "precision_hi": float(np.nanquantile(boot[key + "_prec"], 0.975))}

    # --- robustness: freeze the EV policy (decided under the default assumptions), then let reality differ ---
    sens_rows = []
    for sr in [0.02, 0.05, 0.10, 0.15, 0.20]:
        for cc in [20_000, 35_000, 50_000, 75_000]:
            e = econ.with_(save_rate=sr, contact_cost=cc)
            sens_rows.append({"true_save_rate": sr, "true_contact_cost": cc,
                              "ev_policy": net_benefit(ev_mask, y_test, retention_value(split.test["monetary_90d"], e), e) / len(y_test) * 1000,
                              "flat_policy": net_benefit(flat_mask, y_test, retention_value(split.test["monetary_90d"], e), e) / len(y_test) * 1000})
    sensitivity = pd.DataFrame(sens_rows)
    gain_per_unit_save = float((y_test * test_value)[ev_mask].sum())
    break_even_save_rate = float(ev_mask.sum() * econ.contact_cost / gain_per_unit_save)
    break_even_save_rate_flat = float(flat_mask.sum() * econ.contact_cost / float((y_test * test_value)[flat_mask].sum()))

    # --- what the EV rule means as probability thresholds, by customer value quintile ---------------------
    q = pd.qcut(pd.Series(test_value).rank(method="first"), 5, labels=["Q1 (lowest value)", "Q2", "Q3", "Q4", "Q5 (highest value)"])
    ev_by_value = pd.DataFrame({
        "quintile": q, "value": test_value, "break_even_p": np.minimum(econ.contact_cost / (econ.save_rate * test_value), 1.0),
        "contacted": ev_mask, "churned": y_test}).groupby("quintile", observed=True).agg(
        median_value=("value", "median"), median_break_even_p=("break_even_p", "median"),
        share_contacted=("contacted", "mean"), churn_rate=("churned", "mean"))

    return Experiment(
        dataset=ds, panel=panel, split=split, models=models, raw=raw, calibrated=calibrated, calibrators=calibrators,
        selected=selected, val_metrics=val_metrics, test_metrics=test_metrics, test_metrics_uncalibrated=test_raw,
        test_ci=test_ci, paired_auc=paired_auc, leakage=leakage, importance=importance, econ=econ,
        val_sweep=val_sweep, test_sweep=test_sweep, chosen=chosen, policies=policies, policy_ci=policy_ci,
        sensitivity=sensitivity, ev_by_value=ev_by_value, break_even_save_rate=break_even_save_rate, break_even_save_rate_flat=break_even_save_rate_flat, extras={"n_test_snapshots": n_test_snaps, "n_boot": n_boot})
