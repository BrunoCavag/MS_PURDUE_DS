"""
Phase 4 - run the experiments. Results are appended row-by-row (resumable).

Experiment A - benchmark at natural conditions
    python scripts/run_experiments.py A --models lr,xgb,lgbm,cat,mlp --tune --trials 10
    python scripts/run_experiments.py A --models tabpfn1
  Outer CV: the stored stratified splits (default: first repeat = 5 folds; --repeats 3 for all 15).
  Tuned models: Optuna on an inner 80/20 split; for training folds above --hpo-cap rows the
  search runs on a stratified subsample of that size, the final refit uses the full fold.

Experiment B - controlled conditions (n, pi, m)
    python scripts/run_experiments.py B --models lr,xgb,lgbm,cat,mlp,tabpfn1 --reps 3 --n-test 2000
  All models with library defaults + early stopping (no HPO), so that the manipulated data
  factor is the only thing that changes. TFMs are zero-shot by design.

Outputs: results/exp_a/results.csv, results/exp_b/results.csv (+ test predictions for A).
Modern TFMs (TabPFN-3, TabICL) are run with the same script on a GPU (notebooks/colab_tfm_gpu.ipynb).
"""
import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from sml.datasets import CORE, load  # noqa: E402
from sml.metrics import evaluate  # noqa: E402
from sml.models import fit_predict  # noqa: E402
from sml.perturbations import Condition, make_condition  # noqa: E402
from sml.splits import load_splits  # noqa: E402


def done_keys(path, keycols):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(map(tuple, df[keycols].astype(object).where(df[keycols].notna(), "nan")
                   .astype(str).to_numpy()))


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


TFMS = ("tabpfn1", "tabpfn1_mi", "tabpfn", "tabicl")


def run_a(models, datasets, repeats, tune, trials, hpo_cap, out, max_test_tfm=6000,
          max_folds_large=5, trials_large=None):
    path = out / "results.csv"
    keys = done_keys(path, ["dataset", "fold", "model"])
    for name in datasets:
        ds = load(name)
        folds = [f for f in load_splits(name) if f["repeat"] < repeats]
        large = ds.n > 20000
        if large:                                   # CPU budget for Taiwan / GMSC
            folds = folds[:max_folds_large]
        n_trials = trials_large if (large and trials_large) else trials
        for f in folds:
            tr, te_full = np.asarray(f["train"]), np.asarray(f["test"])
            for m in models:
                if (name, str(f["fold"]), m) in keys:
                    continue
                te = te_full
                if m in TFMS and len(te_full) > max_test_tfm:
                    # CPU budget: TFM inference cost is linear in test size. Evaluate on a
                    # fixed stratified subset; analysis re-scores every model on it.
                    te, _ = train_test_split(te_full, train_size=max_test_tfm,
                                             stratify=ds.y.iloc[te_full], random_state=f["fold"])
                    te = np.sort(te)
                try:
                    p, info = fit_predict_capped(m, ds, tr, te, tune, n_trials, hpo_cap,
                                                 seed=f["fold"])
                    met = evaluate(ds.y.iloc[te], p)
                    row = dict(dataset=name, fold=f["fold"], model=m, tuned=tune,
                               n_train=len(tr), n_test=len(te), **met,
                               seconds=info.pop("seconds"), info=json.dumps(info))
                    append(path, row)
                    pdir = out / "preds"
                    pdir.mkdir(exist_ok=True)
                    np.savez_compressed(pdir / f"{name}_f{f['fold']}_{m}.npz",
                                        y=ds.y.iloc[te].to_numpy().astype(np.int8),
                                        p=p.astype(np.float32), idx=te.astype(np.int32))
                    print(f"[A] {name} fold {f['fold']} {m:8s} AUC={met['auc']:.4f} "
                          f"Brier={met['brier']:.4f} ({row['seconds']}s)", flush=True)
                except Exception:  # noqa: BLE001
                    print(f"[A] FAIL {name} {f['fold']} {m}\n{traceback.format_exc()}", flush=True)


def fit_predict_capped(m, ds, tr, te, tune, trials, hpo_cap, seed):
    """Tune on <= hpo_cap rows (stratified subsample), then refit on the full fold."""
    X, y = ds.X, ds.y.to_numpy()
    sent = ds.name == "gmsc"
    if tune and ds.n > 20000 and len(tr) > hpo_cap and m in ("lr", "mlp", "xgb", "lgbm", "cat"):
        sub, _ = train_test_split(tr, train_size=hpo_cap, stratify=y[tr], random_state=seed)
        _, info = fit_predict(m, X.iloc[sub], y[sub], X.iloc[te[:10]], ds.num_cols,
                              ds.cat_cols, tune=True, n_trials=trials, seed=seed,
                              sentinels=sent)
        from sml import models as M
        best = {k: v for k, v in info.items() if k != "seconds"}
        t0 = time.perf_counter()
        p = M.refit_with_params(m, X.iloc[tr], y[tr], X.iloc[te], ds.num_cols, ds.cat_cols,
                                best, seed, sent)
        best["seconds"] = round(info["seconds"] + time.perf_counter() - t0, 3)
        best["hpo_subsample"] = hpo_cap
        return p, best
    return fit_predict(m, X.iloc[tr], y[tr], X.iloc[te], ds.num_cols, ds.cat_cols,
                       tune=tune, n_trials=trials, seed=seed, sentinels=sent)


def run_b(models, datasets, reps, n_test, out, axes_only=False, min_defaults=50):
    design = pd.read_csv(ROOT / "data" / "processed" / "experiment_design.csv")
    design = design[design.dataset.isin(datasets) & (design.rep < reps)]
    if axes_only:   # drop the n x pi interaction grid, keep the three one-factor axes
        bn = design[design.m_noise > 0].groupby("dataset").n_train.first()
        b = design.dataset.map(bn)
        design = design[(design.pi.isna() & (design.m_noise == 0))
                        | ((design.n_train == b) & (design.m_noise == 0))
                        | ((design.n_train == b) & design.pi.isna())]
    path = out / "results.csv"
    keycols = ["dataset", "n_train", "pi", "m_noise", "rep", "model"]
    keys = done_keys(path, keycols)
    cache = {}
    for _, r in design.iterrows():
        pi = None if pd.isna(r.pi) else float(r.pi)
        # low default rates need a larger test set for stable metrics (>= 50 defaults)
        nt = n_test if pi is None else max(n_test, int(np.ceil(min_defaults / pi)))
        c = Condition(r.dataset, int(r.n_train), pi, float(r.m_noise), nt, int(r.rep))
        pi_key = "nan" if pd.isna(r.pi) else str(r.pi)
        todo = [m for m in models
                if (r.dataset, str(c.n_train), pi_key, str(c.m_noise), str(c.rep), m)
                not in keys]
        if not todo:
            continue
        ds = cache.setdefault(r.dataset, load(r.dataset))
        Xtr, ytr, Xte, yte, nc, cc = make_condition(ds, c)
        for m in todo:
            try:
                p, info = fit_predict(m, Xtr, ytr, Xte, nc, cc, tune=False,
                                      seed=c.rep, sentinels=ds.name == "gmsc")
                met = evaluate(yte, p)
                append(path, dict(dataset=r.dataset, n_train=c.n_train, pi=r.pi,
                                  pi_eff=round(float(ytr.mean()), 4), m_noise=c.m_noise,
                                  d_total=Xtr.shape[1], rep=c.rep, model=m, n_test=len(yte),
                                  **met, seconds=info.pop("seconds")))
                print(f"[B] {r.dataset} n={c.n_train} pi={r.pi} m={c.m_noise} r={c.rep} "
                      f"{m:8s} AUC={met['auc']:.4f}", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[B] FAIL {c} {m}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["A", "B"])
    ap.add_argument("--models", default="lr,xgb,lgbm,cat,mlp")
    ap.add_argument("--datasets", default=",".join(CORE))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--hpo-cap", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--max-test-tfm", type=int, default=6000)
    ap.add_argument("--max-folds-large", type=int, default=5)
    ap.add_argument("--trials-large", type=int, default=None)
    ap.add_argument("--axes-only", action="store_true")
    ap.add_argument("--min-defaults", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models, dsets = a.models.split(","), a.datasets.split(",")
    out = Path(a.out) if a.out else ROOT / "results" / ("exp_a" if a.exp == "A" else "exp_b")
    if a.exp == "A":
        run_a(models, dsets, a.repeats, a.tune, a.trials, a.hpo_cap, out, a.max_test_tfm,
              a.max_folds_large, a.trials_large)
    else:
        run_b(models, dsets, a.reps, a.n_test, out, a.axes_only, a.min_defaults)
