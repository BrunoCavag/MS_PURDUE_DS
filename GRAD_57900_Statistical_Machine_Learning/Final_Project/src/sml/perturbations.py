"""
Controlled experimental conditions (Experiment B).

Research question: under which conditions do tabular foundation models (TFMs)
match or beat GBDTs for default prediction? We manipulate three factors
*within* a dataset, so that differences are attributable to the factor and not
to everything else that differs between datasets:

  n      : training-set size                     (sample-size axis)
  pi     : default rate in train AND test         (class-imbalance axis,
           i.e. a simulated low-default portfolio)
  m      : number of uninformative features added, as a multiple of the
           original dimensionality d              (dimensionality axis)

Uninformative features are row-permuted copies of real columns: they keep the
exact marginal distribution (incl. categoricals and NaNs) but carry no signal.
This follows the "uninformative features" manipulation in Grinsztajn et al.
(2022).

The test set is drawn first, has a fixed size and the same prior pi as the
training set, so metrics are comparable across n.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .datasets import CreditDataset


@dataclass(frozen=True)
class Condition:
    dataset: str
    n_train: int
    pi: float | None = None      # None = natural default rate
    m_noise: float = 0.0         # added features = round(m_noise * d)
    n_test: int = 5000
    rep: int = 0

    def as_dict(self):
        return asdict(self)


def _stratified_take(idx_pos, idx_neg, n, pi, rng):
    n_pos = int(round(n * pi))
    n_neg = n - n_pos
    if n_pos > len(idx_pos) or n_neg > len(idx_neg):
        raise ValueError(f"infeasible: need {n_pos} pos/{n_neg} neg, "
                         f"have {len(idx_pos)}/{len(idx_neg)}")
    pos = rng.choice(idx_pos, n_pos, replace=False)
    neg = rng.choice(idx_neg, n_neg, replace=False)
    return pos, neg


def is_feasible(ds: CreditDataset, c: Condition) -> bool:
    pi = ds.default_rate if c.pi is None else c.pi
    n_test = min(c.n_test, ds.n // 3)
    need_pos = round(c.n_train * pi) + round(n_test * pi)
    need_neg = (c.n_train + n_test) - need_pos
    have_pos = int(ds.y.sum())
    return need_pos <= have_pos and need_neg <= ds.n - have_pos and need_pos >= 10


def add_uninformative_features(X: pd.DataFrame, k: int, rng) -> pd.DataFrame:
    if k <= 0:
        return X
    src = rng.choice(X.columns, size=k, replace=True)
    noise = {}
    for j, c in enumerate(src):
        perm = rng.permutation(len(X))
        col = X[c].iloc[perm].reset_index(drop=True)
        noise[f"noise_{j:03d}__{c}"] = col
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(noise)], axis=1)


def make_condition(ds: CreditDataset, c: Condition, base_seed: int = 0):
    """Return (X_train, y_train, X_test, y_test, num_cols, cat_cols)."""
    rng = np.random.default_rng([base_seed, c.rep, c.n_train,
                                 int(1e4 * (c.pi or 0)), int(10 * c.m_noise)])
    pi = ds.default_rate if c.pi is None else c.pi
    y = ds.y.to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    n_test = min(c.n_test, ds.n // 3)
    te_pos, te_neg = _stratified_take(pos, neg, n_test, pi, rng)
    pos_left = np.setdiff1d(pos, te_pos)
    neg_left = np.setdiff1d(neg, te_neg)
    tr_pos, tr_neg = _stratified_take(pos_left, neg_left, c.n_train, pi, rng)

    tr = rng.permutation(np.concatenate([tr_pos, tr_neg]))
    te = rng.permutation(np.concatenate([te_pos, te_neg]))

    X_tr, X_te = ds.X.iloc[tr].reset_index(drop=True), ds.X.iloc[te].reset_index(drop=True)
    k = int(round(c.m_noise * ds.d))
    if k:
        # same source columns for train/test, independent row permutations
        seed_cols = rng.integers(1 << 31)
        X_tr = add_uninformative_features(X_tr, k, np.random.default_rng(seed_cols))
        X_te = add_uninformative_features(X_te, k, np.random.default_rng(seed_cols))
    noise_cols = [c_ for c_ in X_tr.columns if c_.startswith("noise_")]
    num_cols = ds.num_cols + [c_ for c_ in noise_cols if c_.split("__", 1)[1] in ds.num_cols]
    cat_cols = ds.cat_cols + [c_ for c_ in noise_cols if c_.split("__", 1)[1] in ds.cat_cols]
    return (X_tr, y[tr], X_te, y[te], num_cols, cat_cols)


# --------------------------------------------------------------------------- #
# Experimental design (one-factor-at-a-time around a base point + n x pi grid)
# --------------------------------------------------------------------------- #
N_GRID = [250, 500, 1000, 2500, 5000, 10000, 20000]
PI_GRID = [None, 0.10, 0.05, 0.02, 0.01]
M_GRID = [0.0, 1.0, 3.0]
BASE = dict(n_train=2500, pi=None, m_noise=0.0)
N_REPS = 5
GRID_DATASETS = ["taiwan", "gmsc"]   # large enough for the full n x pi grid


def design(datasets: dict[str, CreditDataset]) -> list[Condition]:
    conds: set[Condition] = set()
    for name, ds in datasets.items():
        n_test = min(5000, ds.n // 3)
        n_base = min(BASE["n_train"], int(0.9 * (ds.n - n_test)))  # small sets: use what fits
        for r in range(N_REPS):
            for n in N_GRID:                                   # axis 1: n
                conds.add(Condition(name, n, BASE["pi"], BASE["m_noise"], rep=r))
            for pi in PI_GRID:                                 # axis 2: pi
                conds.add(Condition(name, n_base, pi, BASE["m_noise"], rep=r))
            for m in M_GRID:                                   # axis 3: noise
                conds.add(Condition(name, n_base, BASE["pi"], m, rep=r))
            if name in GRID_DATASETS:                          # interaction n x pi
                for n in N_GRID:
                    for pi in PI_GRID:
                        conds.add(Condition(name, n, pi, 0.0, rep=r))
    feas = [c for c in conds if is_feasible(datasets[c.dataset], c)]
    return sorted(feas, key=lambda c: (c.dataset, c.rep, c.n_train, c.pi or 1, c.m_noise))
