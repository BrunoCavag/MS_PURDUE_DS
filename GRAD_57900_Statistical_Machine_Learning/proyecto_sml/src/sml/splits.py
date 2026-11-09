"""
Reproducible splits for Experiment A (benchmark at natural conditions).

Protocol: repeated stratified K-fold (default 5 folds x 3 repeats = 15 test
folds per dataset). Inside each training fold, Phase 3 carves out an inner
stratified 20% validation split for hyper-parameter tuning / early stopping,
so the outer test fold is never touched during model selection.

Indices are written to data/splits/<dataset>.json.gz so that every model sees
exactly the same partitions (paired comparisons -> valid Wilcoxon/Friedman).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .datasets import ROOT, CreditDataset

SPLIT_DIR = ROOT / "data" / "splits"
SEED = 20261217  # submission date, easy to remember


def make_outer_splits(ds: CreditDataset, n_splits=5, n_repeats=3, seed=SEED) -> list[dict]:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros(ds.n), ds.y)):
        out.append(dict(fold=i, repeat=i // n_splits, train=tr.tolist(), test=te.tolist()))
    return out


def inner_validation_split(train_idx, y, val_size=0.2, seed=SEED):
    """Stratified inner split of an outer training fold (for HPO / early stopping)."""
    train_idx = np.asarray(train_idx)
    tr, va = train_test_split(train_idx, test_size=val_size, random_state=seed,
                              stratify=np.asarray(y)[train_idx])
    return tr, va


def save_splits(ds: CreditDataset, splits: list[dict]) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"{ds.name}.json.gz"
    payload = dict(dataset=ds.name, n=ds.n, seed=SEED,
                   n_splits=max(s["fold"] for s in splits) + 1, folds=splits)
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)
    return path


def load_splits(name: str) -> list[dict]:
    with gzip.open(SPLIT_DIR / f"{name}.json.gz", "rt") as f:
        return json.load(f)["folds"]
