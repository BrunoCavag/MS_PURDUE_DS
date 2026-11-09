"""Exploratory data analysis: one summary table + three figures for the report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .datasets import ROOT, CreditDataset

OUT = ROOT / "results" / "eda"
COLORS = {"australian": "#4C72B0", "german": "#DD8452", "hmeq": "#55A868",
          "taiwan": "#C44E52", "gmsc": "#8172B3", "homecredit": "#937860"}


def summarize(ds: CreditDataset) -> dict:
    X = ds.X
    miss_cell = float(X.isna().to_numpy().mean())
    miss_cols = int((X.isna().sum() > 0).sum())
    max_card = int(max([X[c].nunique() for c in ds.cat_cols], default=0))
    dup = int(pd.concat([X, ds.y], axis=1).duplicated().sum())
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    pos = int(ds.y.sum())
    return dict(dataset=ds.name, n=ds.n, d=ds.d, n_num=len(ds.num_cols),
                n_cat=len(ds.cat_cols), max_cat_cardinality=max_card,
                default_rate=round(ds.default_rate, 4), n_defaults=pos,
                imbalance_ratio=round((ds.n - pos) / pos, 2),
                missing_cells_pct=round(100 * miss_cell, 2), cols_with_missing=miss_cols,
                duplicate_rows=dup, constant_cols=len(const),
                tfm_regime=_regime(ds.n, ds.d))


def _regime(n, d):
    """Where the dataset falls relative to published TFM context limits
    (TabPFNv2 ~10k x 500; TabPFN-2.5 ~50k x 2000; TabICL pretrained up to 60k)."""
    if n <= 10_000:
        return "within TabPFNv2 limit"
    if n <= 50_000:
        return "within TabPFN-2.5 limit"
    return "above 50k: subsample/context"


def summary_table(datasets: dict[str, CreditDataset]) -> pd.DataFrame:
    df = pd.DataFrame([summarize(ds) for ds in datasets.values()]).sort_values("n")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dataset_summary.csv", index=False)
    return df


def fig_landscape(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for _, r in summary.iterrows():
        ax.scatter(r.n, r.default_rate * 100, s=40 + 12 * r.d, color=COLORS.get(r.dataset),
                   alpha=0.85, edgecolor="k", linewidth=0.6)
        ax.annotate(f"{r.dataset}\n(d={r.d})", (r.n, r.default_rate * 100),
                    textcoords="offset points", xytext=(9, -4), fontsize=8)
    for x, lab in [(10_000, "TabPFNv2 limit"), (50_000, "TabPFN-2.5 limit")]:
        ax.axvline(x, ls="--", lw=0.8, color="grey")
        ax.text(x * 1.05, 2, lab, rotation=90, fontsize=7, color="grey", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Number of observations n (log scale)")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Benchmark datasets: size vs. class balance (marker size ∝ d)")
    ax.set_xlim(400, 4e5)
    ax.set_ylim(0, 65)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_dataset_landscape.png", dpi=200)
    plt.close(fig)


def fig_missingness(datasets: dict[str, CreditDataset]):
    rows = []
    for ds in datasets.values():
        m = ds.X.isna().mean()
        for c, v in m[m > 0].items():
            rows.append((ds.name, c, 100 * v))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["dataset", "feature", "pct"]).sort_values("pct")
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(df) + 1))
    ax.barh(df.dataset + " · " + df.feature, df.pct,
            color=[COLORS.get(d) for d in df.dataset])
    ax.set_xlabel("% missing")
    ax.set_title("Features with missing values (raw data)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_missingness.png", dpi=200)
    plt.close(fig)


def fig_univariate_signal(datasets: dict[str, CreditDataset]):
    """Univariate AUC of each feature (|AUC-0.5|+0.5) - a quick view of how much
    signal is spread across features; helps interpret the noise-feature axis."""
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, ds in enumerate(datasets.values()):
        aucs = []
        for c in ds.X.columns:
            x = ds.X[c]
            x = x.cat.codes.astype(float).replace(-1, np.nan) if c in ds.cat_cols else x
            mask = x.notna()
            if x[mask].nunique() < 2:
                continue
            a = roc_auc_score(ds.y[mask], x[mask])
            aucs.append(max(a, 1 - a))
        aucs = np.sort(aucs)[::-1]
        ax.plot(np.arange(1, len(aucs) + 1), aucs, marker="o", ms=3,
                color=COLORS.get(ds.name), label=ds.name)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Univariate AUC (folded)")
    ax.set_title("Univariate signal per feature")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_univariate_signal.png", dpi=200)
    plt.close(fig)


def category_audit(datasets: dict[str, CreditDataset]) -> pd.DataFrame:
    rows = []
    for ds in datasets.values():
        for c in ds.cat_cols:
            vc = ds.X[c].value_counts(dropna=False)
            rows.append(dict(dataset=ds.name, feature=c, levels=len(vc),
                             rarest_level=str(vc.index[-1]), rarest_count=int(vc.iloc[-1])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "categorical_audit.csv", index=False)
    return df
