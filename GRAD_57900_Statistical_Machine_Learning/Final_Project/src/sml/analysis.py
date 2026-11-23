"""
Phase 4 analysis: tables, statistical tests and figures for Experiments A and B.

Experiment A (natural conditions, tuned baselines)
  * per-dataset mean +/- sd over outer folds
  * Friedman test over datasets and over dataset-folds (blocks), Nemenyi CD diagram
  * pairwise Wilcoxon signed-rank (fold level) with Holm correction
  * reliability diagrams
  For datasets where the TFM was scored on a stratified test subset (CPU budget), every
  model is re-scored on exactly the same subset so comparisons stay paired.

Experiment B (controlled conditions, default hyper-parameters)
  * metric vs n, vs pi, vs m
  * Delta = TFM - GBDT (mean of XGB/LGBM/CatBoost, and best-of-three) per condition
  * OLS of Delta on log10 n, log10 pi, m with dataset fixed effects, SE clustered by dataset-rep
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .datasets import ROOT
from .metrics import HIGHER_IS_BETTER, evaluate
from .models import DISPLAY, GROUP
from .splits import load_splits

RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
MODEL_ORDER = ["lr", "mlp", "xgb", "lgbm", "cat", "tabpfn1", "tabpfn1_mi", "tabpfn", "tabicl"]
COL = {"lr": "#7f7f7f", "mlp": "#bcbd22", "xgb": "#1f77b4", "lgbm": "#17becf",
       "cat": "#2ca02c", "tabpfn1": "#d62728", "tabpfn1_mi": "#ff9896", "tabpfn": "#9467bd", "tabicl": "#e377c2"}
DS_ORDER = ["australian", "german", "hmeq", "taiwan", "gmsc", "homecredit"]
GBDT = ["xgb", "lgbm", "cat"]
TFM = ["tabpfn1", "tabpfn", "tabicl"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110})


def _order(models):
    return [m for m in MODEL_ORDER if m in set(models)]


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== Exp A
def load_exp_a_paired(exp_dir=RES / "exp_a") -> pd.DataFrame:
    """Re-score all models on a common test set per (dataset, fold)."""
    pdir = exp_dir / "preds"
    files = sorted(pdir.glob("*.npz"))
    recs = {}
    for f in files:
        import re
        ds, fold, model = re.match(r"^(.+?)_f(\d+)_(.+)$", f.stem).groups()
        fold = int(fold)
        recs[(ds, fold, model)] = f
    splits_cache = {}
    rows = []
    for (ds, fold) in sorted({(k[0], k[1]) for k in recs}):
        if ds not in splits_cache:
            splits_cache[ds] = {s["fold"]: np.asarray(s["test"]) for s in load_splits(ds)}
        full = splits_cache[ds][fold]
        preds = {}
        for m in MODEL_ORDER:
            if (ds, fold, m) not in recs:
                continue
            z = np.load(recs[(ds, fold, m)])
            idx = z["idx"] if "idx" in z.files else full
            preds[m] = pd.Series(z["p"], index=idx), pd.Series(z["y"], index=idx)
        common = None
        for p, _ in preds.values():
            common = p.index if common is None else common.intersection(p.index)
        for m, (p, y) in preds.items():
            met = evaluate(y.loc[common].to_numpy(), p.loc[common].to_numpy())
            rows.append(dict(dataset=ds, fold=fold, model=m, n_test=len(common),
                             subset=len(common) < len(full), **met))
    return pd.DataFrame(rows)


def table_a(df: pd.DataFrame, metrics=("auc", "brier", "ece", "ks", "pr_auc")) -> pd.DataFrame:
    g = df.groupby(["dataset", "model"])[list(metrics)].agg(["mean", "std"])
    TAB.mkdir(parents=True, exist_ok=True)
    g.to_csv(TAB / "expA_metrics_mean_sd.csv")
    return g


def pretty_table_a(df, metric="auc", digits=3):
    m = df.groupby(["dataset", "model"])[metric].agg(["mean", "std"]).reset_index()
    m["cell"] = m.apply(lambda r: f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}", axis=1)
    t = m.pivot(index="dataset", columns="model", values="cell")
    t = t.loc[[d for d in DS_ORDER if d in t.index], _order(t.columns)]
    t.columns = [DISPLAY[c] for c in t.columns]
    return t


def ranks(df, metric="auc", level="dataset"):
    """Average rank per model (1 = best). level='dataset' averages folds first."""
    hib = HIGHER_IS_BETTER[metric]
    if level == "dataset":
        piv = df.groupby(["dataset", "model"])[metric].mean().unstack()
    else:
        piv = df.pivot_table(index=["dataset", "fold"], columns="model", values=metric)
    piv = piv.dropna(axis=0)
    r = piv.rank(axis=1, ascending=not hib)
    return piv, r


def friedman(df, metric="auc", level="dataset"):
    piv, r = ranks(df, metric, level)
    stat, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    N, k = piv.shape
    # Iman-Davenport correction
    ff = (N - 1) * stat / (N * (k - 1) - stat) if N * (k - 1) != stat else np.inf
    p_ff = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (N - 1))
    return dict(metric=metric, level=level, N=N, k=k, chi2=stat, p_chi2=p, F_ID=ff, p_ID=p_ff,
                avg_rank=r.mean().sort_values().round(2).to_dict())


def wilcoxon_holm(df, metric="auc", ref=None):
    """Fold-level paired Wilcoxon of every model vs `ref` (default: best average), Holm."""
    piv = df.pivot_table(index=["dataset", "fold"], columns="model", values=metric).dropna()
    hib = HIGHER_IS_BETTER[metric]
    if ref is None:
        ref = (piv.mean().idxmax() if hib else piv.mean().idxmin())
    rows = []
    for m in piv.columns:
        if m == ref:
            continue
        d = piv[ref] - piv[m]
        w = stats.wilcoxon(d, zero_method="zsplit")
        rows.append(dict(ref=ref, model=m, median_diff=float(np.median(d)),
                         wins_ref=int(((d > 0) if hib else (d < 0)).sum()), n=len(d),
                         p=w.pvalue))
    out = pd.DataFrame(rows).sort_values("p")
    k = len(out)
    out["p_holm"] = np.maximum.accumulate((out["p"].to_numpy() * (k - np.arange(k)))).clip(max=1)
    return out


def nemenyi_cd(k, N, alpha=0.05):
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
         10: 3.164}[k]  # Demsar (2006), Table 5(a), alpha = 0.05
    return q * np.sqrt(k * (k + 1) / (6 * N))


def fig_cd(df, metric="auc", level="fold", name="figA2_cd_auc.png"):
    piv, r = ranks(df, metric, level)
    avg = r.mean().sort_values()
    k, N = piv.shape[1], piv.shape[0]
    cd = nemenyi_cd(k, N)
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    ax.set_xlim(0.8, k + 0.2)
    ax.set_ylim(0, 1)
    ax.axhline(0.75, color="k", lw=0.8)
    for i in range(1, k + 1):
        ax.plot([i, i], [0.73, 0.77], color="k", lw=0.8)
        ax.text(i, 0.82, str(i), ha="center", fontsize=8)
    for j, (m, v) in enumerate(avg.items()):
        y = 0.6 - 0.11 * (j % 5)
        ax.plot([v, v], [y, 0.75], color=COL[m], lw=1)
        ax.plot(v, 0.75, "o", color=COL[m], ms=4)
        ax.text(v, y - 0.04, f"{DISPLAY[m]} ({v:.2f})", ha="center", va="top", fontsize=7.5,
                color=COL[m])
    ax.plot([1, 1 + cd], [0.95, 0.95], color="k", lw=2)
    ax.text(1 + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axis("off")
    ax.set_title(f"Average rank by {metric.upper()} over {N} dataset-folds "
                 "(1 = best; Nemenyi α=0.05)", fontsize=9)
    _save(fig, name)
    return avg, cd


def fig_a_dots(df, metrics=("auc", "brier"), name="figA1_benchmark.png"):
    dsets = [d for d in DS_ORDER if d in df.dataset.unique()]
    models = _order(df.model.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.0), sharey=True)
    for ax, met in zip(np.atleast_1d(axes), metrics):
        g = df.groupby(["dataset", "model"])[met].agg(["mean", "std"])
        for j, m in enumerate(models):
            ys = np.arange(len(dsets)) + (j - len(models) / 2) * 0.1
            mu = [g.loc[(d, m), "mean"] if (d, m) in g.index else np.nan for d in dsets]
            sd = [g.loc[(d, m), "std"] if (d, m) in g.index else np.nan for d in dsets]
            ax.errorbar(mu, ys, xerr=sd, fmt="o", ms=3.5, color=COL[m], label=DISPLAY[m],
                        elinewidth=0.8, capsize=0)
        ax.set_yticks(range(len(dsets)))
        ax.set_yticklabels(dsets)
        ax.set_xlabel(met.upper() + (" (higher better)" if HIGHER_IS_BETTER[met]
                                      else " (lower better)"))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left", ncol=2, frameon=True)
    fig.suptitle("Experiment A: tuned baselines vs TabPFN at natural conditions "
                 "(mean ± sd over 5 folds)", fontsize=9.5)
    fig.tight_layout()
    _save(fig, name)


def fig_reliability(exp_dir=RES / "exp_a", datasets=("taiwan", "gmsc"),
                    models=("lr", "cat", "lgbm", "tabpfn1"), name="figA3_reliability.png"):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.1))
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        full = {s["fold"]: np.asarray(s["test"]) for s in load_splits(ds)}
        sub = None
        for f in range(5):
            fp = exp_dir / "preds" / f"{ds}_f{f}_tabpfn1.npz"
            if fp.exists():
                z = np.load(fp)
                sub = (sub or {}) | {f: set(z["idx"])}
        for m in models:
            ys, ps = [], []
            for f in range(5):
                fp = exp_dir / "preds" / f"{ds}_f{f}_{m}.npz"
                if not fp.exists():
                    continue
                z = np.load(fp)
                idx = z["idx"] if "idx" in z.files else full[f]
                keep = np.isin(idx, list(sub[f])) if sub and f in sub else np.ones(len(idx), bool)
                ys.append(z["y"][keep])
                ps.append(z["p"][keep])
            if not ys:
                continue
            y, p = np.concatenate(ys), np.concatenate(ps)
            q = np.quantile(p, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
            mp = [p[b == i].mean() for i in range(10)]
            my = [y[b == i].mean() for i in range(10)]
            ax.plot(mp, my, "o-", ms=3, color=COL[m], label=DISPLAY[m])
        lim = ax.get_xlim()[1]
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("Mean predicted PD (decile)")
        ax.set_ylabel("Observed default rate")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Reliability diagrams (pooled outer test folds)", fontsize=9.5)
    fig.tight_layout()
    _save(fig, name)


# =========================================================================== Exp B
def load_exp_b(exp_dir=RES / "exp_b") -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(exp_dir.glob("results*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["pi_label"] = df["pi"].apply(lambda v: "natural" if pd.isna(v) else f"{v:g}")
    df = df.drop_duplicates(["dataset", "n_train", "pi_label", "m_noise", "rep", "model"],
                            keep="last")
    return df


def base_n(df):
    """Base training size per dataset (the n used on the pi and m axes)."""
    sub = df[(df.m_noise > 0)]
    return sub.groupby("dataset").n_train.first().to_dict()


def deltas(df, metric="auc", tfm="tabpfn1"):
    keys = ["dataset", "n_train", "pi_label", "m_noise", "rep"]
    piv = df.pivot_table(index=keys, columns="model", values=metric)
    gb = [c for c in GBDT if c in piv.columns]
    hib = HIGHER_IS_BETTER[metric]
    out = piv.reset_index()
    out["gbdt_mean"] = piv[gb].mean(axis=1).to_numpy()
    out["gbdt_best"] = (piv[gb].max(axis=1) if hib else piv[gb].min(axis=1)).to_numpy()
    sign = 1 if hib else -1
    out["delta_vs_mean"] = sign * (out[tfm] - out["gbdt_mean"])   # >0: TFM better
    out["delta_vs_best"] = sign * (out[tfm] - out["gbdt_best"])
    out["metric"] = metric
    return out


def axis_slice(df, axis):
    bn = base_n(df)
    is_base_n = df.apply(lambda r: r.n_train == bn.get(r.dataset), axis=1)
    if axis == "n":
        return df[(df.pi_label == "natural") & (df.m_noise == 0)]
    if axis == "pi":
        return df[is_base_n & (df.m_noise == 0)]
    if axis == "m":
        return df[is_base_n & (df.pi_label == "natural")]
    raise ValueError(axis)


def fig_b_axis(df, axis, metric="auc", name=None, datasets=None):
    sl = axis_slice(df, axis)
    datasets = datasets or [d for d in DS_ORDER if d in sl.dataset.unique()
                            and sl[sl.dataset == d][{"n": "n_train", "pi": "pi_label",
                                                     "m": "m_noise"}[axis]].nunique() > 1]
    models = _order(sl.model.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets), 2.7),
                             sharey=False)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        s = sl[sl.dataset == ds].copy()
        if axis == "pi":
            s["x"] = s.apply(lambda r: r.pi_eff if r.pi_label == "natural" else float(r.pi_label),
                             axis=1)
        else:
            s["x"] = s.n_train if axis == "n" else s.m_noise
        g = s.groupby(["model", "x"])[metric].agg(["mean", "std"]).reset_index()
        for m in models:
            gm = g[g.model == m].sort_values("x")
            if gm.empty:
                continue
            lw = 2 if m in TFM else 1.1
            ax.plot(gm.x, gm["mean"], "o-", ms=3, lw=lw, color=COL[m], label=DISPLAY[m])
            ax.fill_between(gm.x, gm["mean"] - gm["std"], gm["mean"] + gm["std"],
                            color=COL[m], alpha=0.08)
        if axis in ("n", "pi"):
            from matplotlib.ticker import FuncFormatter
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: (f"{v:.0%}" if v < 1 else f"{int(v):,}")))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel({"n": "training size n", "pi": "default rate π",
                       "m": "noise features (× d)"}[axis])
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(metric.upper())
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(models), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, name or f"figB_{axis}_{metric}.png")


def fig_b_delta_heatmap(df, metric="auc", tfm="tabpfn1", name="figB4_heatmap_n_pi.png"):
    d = deltas(df[(df.m_noise == 0)], metric, tfm)
    dsets = [x for x in ["taiwan", "gmsc"] if x in d.dataset.unique()]
    fig, axes = plt.subplots(1, len(dsets), figsize=(3.4 * len(dsets), 2.8))
    vmax = np.nanpercentile(np.abs(d.delta_vs_mean), 95)
    for ax, ds in zip(np.atleast_1d(axes), dsets):
        s = d[d.dataset == ds].groupby(["pi_label", "n_train"]).delta_vs_mean.mean().unstack()
        order = ["natural", "0.1", "0.05", "0.02", "0.01"]
        s = s.loc[[o for o in order if o in s.index]]
        im = ax.imshow(s.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(s.shape[1]))
        ax.set_xticklabels([f"{c // 1000}k" if c >= 1000 else str(c) for c in s.columns],
                           fontsize=7)
        ax.set_yticks(range(s.shape[0]))
        ax.set_yticklabels(s.index, fontsize=7)
        for i, j in itertools.product(range(s.shape[0]), range(s.shape[1])):
            v = s.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6)
        ax.set_xlabel("training size n")
        ax.set_ylabel("default rate π")
        ax.set_title(ds)
    fig.colorbar(im, ax=axes, shrink=0.8, label=f"Δ{metric.upper()} (TFM − mean GBDT)")
    _save(fig, name)


def regression_b(df, metric="auc", tfm="tabpfn1", target="delta_vs_mean"):
    import statsmodels.formula.api as smf
    d = deltas(df, metric, tfm).dropna(subset=[tfm, "gbdt_mean"])
    d["log10_n"] = np.log10(d.n_train)
    d["log10_pi"] = np.log10(d.apply(lambda r: base_rate(r.dataset) if r.pi_label == "natural"
                                     else float(r.pi_label), axis=1))
    d["cluster"] = d.dataset + "_" + d.rep.astype(str)
    mod = smf.ols(f"{target} ~ log10_n + log10_pi + m_noise + C(dataset)", data=d)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cluster)[0]})
    return res, d


_BR = {}


def base_rate(ds):
    if ds not in _BR:
        from .datasets import load
        _BR[ds] = load(ds).default_rate
    return _BR[ds]


def summary_json(obj, name):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(json.dumps(obj, indent=2, default=float))


DS_COL = {"australian": "#4C72B0", "german": "#DD8452", "hmeq": "#55A868",
          "taiwan": "#C44E52", "gmsc": "#8172B3", "homecredit": "#937860"}


def fig_b_delta_panels(df, metric="auc", tfm="tabpfn1", name=None):
    """Delta = TFM - mean GBDT (sign: >0 means TFM better) along the three axes,
    one line per dataset, mean +/- sd over replicates."""
    from matplotlib.ticker import FuncFormatter
    d = deltas(df, metric, tfm).dropna(subset=[tfm])
    d["model"] = "x"
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    for ax, axis in zip(axes, ("n", "pi", "m")):
        sl = axis_slice(d, axis).copy()
        if axis == "pi":
            sl["x"] = sl.apply(lambda r: base_rate(r.dataset) if r.pi_label == "natural"
                               else float(r.pi_label), axis=1)
        else:
            sl["x"] = sl.n_train if axis == "n" else sl.m_noise
        for ds in [x for x in DS_ORDER if x in sl.dataset.unique()]:
            g = sl[sl.dataset == ds].groupby("x").delta_vs_mean.agg(["mean", "std"]).reset_index()
            if len(g) < 2:
                continue
            ax.errorbar(g.x, g["mean"], yerr=g["std"].fillna(0), fmt="o-", ms=3.5, lw=1.4,
                        capsize=2, color=DS_COL[ds], label=ds)
        ax.axhline(0, color="k", lw=0.8)
        if axis in ("n", "pi"):
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: (f"{v:.0%}" if v < 1 else f"{int(v):,}")))
        ax.set_xlabel({"n": "training size n", "pi": "default rate π (log)",
                       "m": "added noise features (× d)"}[axis])
        ax.set_title({"n": "H1: sample size", "pi": "H2: class imbalance",
                      "m": "H3: uninformative features"}[axis])
        ax.grid(alpha=0.3)
    lab = {"auc": "ΔAUC", "brier": "ΔBrier (sign flipped)", "ece": "ΔECE (sign flipped)"}[metric]
    axes[0].set_ylabel(f"{lab}\nTabPFN − mean GBDT")
    axes[0].text(0.02, 0.97, "TFM better ↑", transform=axes[0].transAxes, fontsize=7.5, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, name or f"figB_delta_{metric}.png")
