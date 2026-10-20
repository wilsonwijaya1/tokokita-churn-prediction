"""Figures. Colour-blind-safe palette (Okabe-Ito subset); one idea per chart; direct labels where possible."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .features import LABEL_COLUMN  # noqa: E402
from .models import MODEL_NAMES  # noqa: E402

INK, MUTED, GRID, SURFACE = "#1f2328", "#59636e", "#e3e6ea", "#ffffff"
BLUE, ORANGE, GREEN, PINK, GRAY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8c959f"
MODEL_COLORS = {"Logistic Regression": GRAY, "Random Forest": GREEN, "XGBoost": BLUE, "LightGBM": PINK}
SPLIT_COLORS = {"train": GRAY, "val": GREEN, "test": BLUE}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.edgecolor": "#c9ced4", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    })


def _fmt_idr(v, _=None):
    return f"{v/1e6:,.1f}M" if abs(v) >= 1e6 else f"{v/1e3:,.0f}k"


def fig_data_overview(exp):
    style()
    tx, panel, split = exp.dataset.transactions, exp.panel, exp.split
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    m = tx.groupby(tx["order_date"].dt.to_period("M").dt.to_timestamp()).size()
    axes[0].plot(m.index, m.values, color=BLUE, lw=2)
    axes[0].set(title="Orders per month (synthetic)", ylabel="orders")
    by = panel.groupby("as_of")[LABEL_COLUMN].mean()
    for name, df in (("train", split.train), ("val", split.val), ("test", split.test)):
        s = by.loc[sorted(set(df["as_of"]))]
        axes[1].plot(s.index, s.values, "o-", color=SPLIT_COLORS[name], lw=1.5, ms=4,
                     label={"train": "train", "val": "validation", "test": "out-of-time test"}[name])
    axes[1].set(title="30-day churn rate by snapshot date", ylabel="share of active customers who churn", ylim=(0.2, 0.45))
    axes[1].legend(loc="lower left", ncols=3, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_leakage(exp):
    style()
    la, ca = exp.leakage["audit_leaky"].head(6), exp.leakage["audit_clean"].head(6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, df, color, title in ((axes[0], la, ORANGE, "Leaky: features computed at end of label window"),
                                 (axes[1], ca, BLUE, "Clean: features computed at the snapshot date")):
        ax.barh(df["feature"][::-1], df["strength"][::-1], color=color, height=0.6)
        for i, v in enumerate(df["strength"][::-1]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=MUTED)
        ax.axvline(0.90, color=INK, lw=1, ls="--")
        ax.set(title=title, xlim=(0.5, 1.08))
        ax.grid(axis="y", visible=False)
    axes[0].text(0.895, 5.45, "audit threshold 0.90", fontsize=8.5, color=INK, va="bottom", ha="right")
    axes[0].set_ylim(-0.6, 5.9)
    fig.supxlabel("single-feature AUC (direction-agnostic)", color=MUTED, fontsize=10)
    fig.tight_layout()
    return fig


def fig_model_comparison(exp):
    style()
    metrics = [("AUC", "ROC AUC"), ("PR_AUC", "PR AUC"), ("Precision@10%", "Precision@10%")]
    ci = exp.test_ci
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, (key, title) in zip(axes, metrics):
        for i, m in enumerate(MODEL_NAMES):
            r = ci[(ci.model == m) & (ci.metric == key)].iloc[0]
            ax.plot([r.lo, r.hi], [i, i], color=MODEL_COLORS[m], lw=2.5, solid_capstyle="round")
            ax.scatter(r.point, i, s=55, color=MODEL_COLORS[m], edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.text(r.point, i - 0.28, f"{r.point:.3f}", ha="center", fontsize=8.5, color=MUTED)
        ax.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES if ax is axes[0] else [""] * 4)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Out-of-time test, point estimate and 95% customer-clustered bootstrap CI", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    return fig


def fig_calibration(exp):
    style()
    sel, y = exp.selected, exp.split.test[LABEL_COLUMN].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, (label, p, color) in zip(axes, (("Raw model output", exp.raw["test"][sel], ORANGE),
                                            ("After isotonic calibration (fit on validation)", exp.calibrated["test"][sel], BLUE))):
        bins = np.clip((p * 10).astype(int), 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = bins == b
            if m.sum() >= 20:
                xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], color=INK, lw=1, ls="--")
        ax.plot(xs, ys, "-", color=color, lw=1.5)
        ax.scatter(xs, ys, s=np.sqrt(ns) * 6, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.set(title=label, xlabel="predicted churn probability", xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
    axes[0].set_ylabel("observed churn rate")
    fig.suptitle(f"Reliability on the out-of-time test set ({sel}); marker size = customers in bin", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    return fig


def fig_lift(exp):
    style()
    sel = exp.selected
    y = exp.split.test[LABEL_COLUMN].to_numpy()
    p = exp.raw["test"][sel]
    d = pd.DataFrame({"p": p, "y": y})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    rate = d.groupby("decile", observed=True)["y"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(1, 11), rate.values, 0.62, color=BLUE)
    ax.axhline(y.mean(), color=INK, lw=1)
    ax.text(10.4, y.mean() + 0.015, f"base rate {y.mean():.1%}", ha="right", fontsize=9)
    for i, v in enumerate(rate.values):
        ax.text(i + 1, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set(title=f"Observed churn rate by risk decile ({sel}, out-of-time test)",
           xlabel="risk decile (1 = highest predicted risk)", ylim=(0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_net_benefit(exp):
    style()
    v, t = exp.val_sweep, exp.test_sweep
    ev_nb = exp.policies.loc[exp.policies.policy.str.startswith("Expected-value"), "net_benefit_per_1000_scored"].iloc[0]
    t_star = float(exp.chosen["threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(v.threshold, v.net_benefit_per_1000_scored, color=GREEN, lw=2, label="validation")
    ax.plot(t.threshold, t.net_benefit_per_1000_scored, color=BLUE, lw=2, label="out-of-time test")
    ax.axhline(0, color=INK, lw=1)
    ax.axhline(ev_nb, color=ORANGE, lw=1.5, ls="--")
    ax.text(0.01, ev_nb + 0.25e6, "value-aware expected-value rule (test)", color=INK, fontsize=9)
    ax.axvline(t_star, color=MUTED, lw=1, ls=":")
    ax.text(t_star - 0.01, -6.4e6, f"best flat threshold\non validation: {t_star:.2f}", ha="right", fontsize=8.5, color=MUTED)
    ax.yaxis.set_major_formatter(_fmt_idr)
    ax.set(title="Net benefit vs flat probability threshold", xlabel="contact customers with calibrated P(churn) >= threshold",
           ylabel="net benefit per 1,000 scored customers (IDR)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=9)
    ax = axes[1]
    ax.plot(t.threshold, t.share_contacted, color=BLUE, lw=2, label="share contacted")
    ax.plot(t.threshold, t.precision, color=ORANGE, lw=2, label="precision of contacted")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set(title="Who gets contacted (test)", xlabel="threshold", ylim=(0, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=9)
    fig.tight_layout()
    return fig


def fig_sensitivity(exp):
    style()
    s = exp.sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f2f2", BLUE])
    for ax, col, title in ((axes[0], "ev_policy", "Value-aware expected-value rule"),
                           (axes[1], "flat_policy", f"Flat threshold p >= {float(exp.chosen['threshold']):.2f}")):
        pv = s.pivot(index="true_save_rate", columns="true_contact_cost", values=col)
        lim = max(abs(pv.values).max(), 1.0)
        ax.imshow(pv.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
        ax.set_xticks(range(pv.shape[1]), [f"{c/1e3:.0f}k" for c in pv.columns])
        ax.set_yticks(range(pv.shape[0]), [f"{r:.0%}" for r in pv.index])
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, _fmt_idr(pv.values[i, j]), ha="center", va="center", fontsize=8.5, color=INK)
        ax.grid(False)
        ax.set(title=title, xlabel="actual cost per contact (IDR)")
    axes[0].set_ylabel("actual save rate")
    fig.suptitle("Net benefit per 1,000 scored customers when reality differs from the planning assumptions "
                 "(policy frozen; blue = profit, orange = loss)", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    return fig


def fig_importance(exp):
    style()
    imp = exp.importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(imp["feature"], imp["auc_drop"], xerr=imp["std"], color=BLUE, height=0.6, error_kw=dict(ecolor=MUTED, lw=1))
    ax.set(title=f"Permutation importance on validation ({exp.selected})", xlabel="drop in ROC AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


FIGURES = {
    "01_data_overview": fig_data_overview,
    "02_leakage_audit": fig_leakage,
    "03_model_comparison_oot": fig_model_comparison,
    "04_calibration": fig_calibration,
    "05_lift_by_decile": fig_lift,
    "06_net_benefit_vs_threshold": fig_net_benefit,
    "07_sensitivity": fig_sensitivity,
    "08_feature_importance": fig_importance,
}


def save_all(exp, outdir: str | Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        fig = fn(exp)
        p = outdir / f"{name}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
