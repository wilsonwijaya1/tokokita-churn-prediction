"""Run the whole experiment headlessly: data -> models -> figures -> reports/results.json.

Usage:  python scripts/run_pipeline.py [--n-boot 1000] [--save-data]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokokita import config as C  # noqa: E402
from tokokita.data import save_dataset  # noqa: E402
from tokokita.plots import save_all  # noqa: E402
from tokokita.pipeline import run_experiment  # noqa: E402


def summarise(exp) -> dict:
    s = exp.split
    base = {
        "seed": C.SEED, "n_customers": int(exp.dataset.customers.shape[0]), "n_orders": int(exp.dataset.transactions.shape[0]),
        "panel_rows": int(exp.panel.shape[0]),
        "rows": {"train": len(s.train), "val": len(s.val), "test": len(s.test)},
        "churn_rate": {k: float(d["churn_30d"].mean()) for k, d in (("train", s.train), ("val", s.val), ("test", s.test))},
        "test_snapshots": [str(d.date()) for d in sorted(set(s.test["as_of"]))],
        "selected_model_by_val_auc": exp.selected,
        "val_metrics": exp.val_metrics.round(4).to_dict("index"),
        "test_metrics": exp.test_metrics.round(4).to_dict("index"),
        "test_metrics_uncalibrated": exp.test_metrics_uncalibrated.round(4).to_dict("index"),
        "test_ci": exp.test_ci.round(4).to_dict("records"),
        "paired_auc_vs_selected": exp.paired_auc.round(4).reset_index().to_dict("records"),
        "leakage": {
            "recency_auc_leaky": round(exp.leakage["recency_auc_leaky"], 4),
            "recency_auc_clean": round(exp.leakage["recency_auc_clean"], 4),
            "leaky_eval_metrics": exp.leakage["leaky_metrics"].round(4).to_dict("index"),
            "flagged_leaky": exp.leakage["audit_leaky"].query("strength >= 0.9")["feature"].tolist(),
            "flagged_clean": exp.leakage["audit_clean"].query("strength >= 0.9")["feature"].tolist(),
            "max_single_feature_auc_clean": round(float(exp.leakage["audit_clean"]["strength"].max()), 4),
        },
        "top_features": exp.importance.head(10).round(4).to_dict("records"),
        "economics": {
            "assumptions": exp.econ.__dict__,
            "chosen_flat_threshold": float(exp.chosen["threshold"]),
            "policies": exp.policies.round(4).to_dict("records"),
            "policy_ci": exp.policy_ci,
            "ev_by_value_quintile": exp.ev_by_value.round(4).reset_index().to_dict("records"),
            "break_even_save_rate_ev": round(exp.break_even_save_rate, 4),
            "break_even_save_rate_flat": round(exp.break_even_save_rate_flat, 4),
            "sensitivity": exp.sensitivity.round(0).to_dict("records"),
            "n_test_snapshots": exp.extras["n_test_snapshots"],
        },
        "n_boot": exp.extras["n_boot"],
    }
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--save-data", action="store_true", help="also write the synthetic CSVs to data/")
    args = ap.parse_args()

    t0 = time.time()
    exp = run_experiment(n_boot=args.n_boot, verbose=True)
    figs = save_all(exp, ROOT / "reports" / "figures")
    if args.save_data:
        save_dataset(exp.dataset, ROOT / "data")
    out = ROOT / "reports" / "results.json"
    out.write_text(json.dumps(summarise(exp), indent=2, default=float), encoding="utf-8")
    print(f"wrote {out} and {len(figs)} figures in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
