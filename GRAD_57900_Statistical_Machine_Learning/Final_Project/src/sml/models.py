"""
Model registry for Phase 3.

Every model exposes the same function

    fit_predict(name, X_tr, y_tr, X_te, num_cols, cat_cols, tune, n_trials, seed)
        -> (p_test, info)

* Preprocessing is fitted inside this call on X_tr only (see preprocessing.py).
* Tuned models: Optuna TPE on an inner stratified 80/20 split of the training
  data (maximise validation AUC); the final model is refitted on 100% of the
  training data (GBDTs use 1.25 x the early-stopped number of trees).
* Default models: library defaults + early stopping on the same inner split.
* TFMs are never tuned (zero-shot in-context learning by design).

Models available in this repo
-----------------------------
lr        logistic regression (L2)                         family linear
xgb       XGBoost (hist)                                   family tree
lgbm      LightGBM (native categoricals)                   family tree
cat       CatBoost (native categoricals)                   family tree
mlp       multilayer perceptron (sklearn, early stopping)  family nn
tabpfn1   TabPFN v1 (ICLR 2023), CPU; context <= 1000 rows -> bagged contexts
tabpfn    TabPFN v2 / 2.5 / 3 via `tabpfn` package  (GPU; Colab notebook)
tabicl    TabICL via `tabicl` package                 (GPU; Colab notebook)
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .preprocessing import _prep_categoricals_for_sklearn, make_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FAMILY = {"tabpfn1_mi": "tfm", "lr": "linear", "xgb": "tree", "lgbm": "tree", "cat": "tree", "mlp": "nn",
          "tabpfn1": "tfm", "tabpfn": "tfm", "tabicl": "tfm"}
DISPLAY = {"tabpfn1_mi": "TabPFN-v1+MI", "lr": "LogReg", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost",
           "mlp": "MLP", "tabpfn1": "TabPFN-v1", "tabpfn": "TabPFN-3", "tabicl": "TabICL"}
GROUP = {"tabpfn1_mi": "TFM", "lr": "Linear", "xgb": "GBDT", "lgbm": "GBDT", "cat": "GBDT", "mlp": "NN",
         "tabpfn1": "TFM", "tabpfn": "TFM", "tabicl": "TFM"}
N_JOBS = int(os.environ.get("SML_N_JOBS", "1"))
ES_ROUNDS = 50
MAX_TREES = 2000


# --------------------------------------------------------------------------- utils
def _preprocess(family, X_tr, X_te, num_cols, cat_cols, sentinels):
    pre = make_preprocessor(family, num_cols, cat_cols, handle_sentinels=sentinels)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        A = pre.fit_transform(_prep_categoricals_for_sklearn(X_tr, cat_cols))
        B = pre.transform(_prep_categoricals_for_sklearn(X_te, cat_cols))
    return A, B


def _as_category(df, cat_cols, ref=None):
    """Ordinal codes -> pandas category with integer categories. `ref` (the training
    frame) fixes the category set so that train and test share the same encoding."""
    df = df.copy()
    for c in cat_cols:
        src = df[c] if ref is None else ref[c]
        cats = sorted(set(src.dropna().astype(int)) | {-1})
        df[c] = pd.Categorical(df[c].round().astype("Int64"), categories=cats)
    return df


def _inner_split(n, y, seed):
    idx = np.arange(n)
    return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)


def _optimize(objective, n_trials, seed):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_trial.user_attrs


# --------------------------------------------------------------------------- linear
def _lr(A, y, B, tune, n_trials, seed):
    from sklearn.linear_model import LogisticRegression

    def make(C):
        return LogisticRegression(C=C, max_iter=2000, solver="lbfgs")

    C = 1.0
    if tune:
        tr, va = _inner_split(len(y), y, seed)

        def obj(t):
            m = make(t.suggest_float("C", 1e-3, 1e2, log=True)).fit(A.iloc[tr], y[tr])
            return roc_auc_score(y[va], m.predict_proba(A.iloc[va])[:, 1])
        C = _optimize(obj, n_trials, seed)[0]["C"]
    m = make(C).fit(A, y)
    return m.predict_proba(B)[:, 1], {"C": C}


# --------------------------------------------------------------------------- MLP
def _mlp(A, y, B, tune, n_trials, seed):
    from sklearn.neural_network import MLPClassifier
    sizes = {"64": (64,), "128-64": (128, 64), "256-128": (256, 128)}

    def make(h, alpha, lr):
        return MLPClassifier(hidden_layer_sizes=sizes[h], alpha=alpha, learning_rate_init=lr,
                             early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=15, max_iter=300, batch_size=256,
                             random_state=seed)

    params = dict(h="128-64", alpha=1e-4, lr=1e-3)
    if tune:
        tr, va = _inner_split(len(y), y, seed)

        def obj(t):
            p = dict(h=t.suggest_categorical("h", list(sizes)),
                     alpha=t.suggest_float("alpha", 1e-6, 1e-1, log=True),
                     lr=t.suggest_float("lr", 1e-4, 1e-2, log=True))
            m = make(**p).fit(A.iloc[tr], y[tr])
            return roc_auc_score(y[va], m.predict_proba(A.iloc[va])[:, 1])
        params = _optimize(obj, n_trials, seed)[0]
    m = make(**params).fit(A, y)
    return m.predict_proba(B)[:, 1], params


# --------------------------------------------------------------------------- GBDTs
def _xgb(A, y, B, cat_cols, tune, n_trials, seed):
    import xgboost as xgb
    A, B = _as_category(A, cat_cols), _as_category(B, cat_cols, ref=A)
    tr, va = _inner_split(len(y), y, seed)

    def fit(p, Xa, ya, Xv=None, yv=None, n_est=MAX_TREES):
        m = xgb.XGBClassifier(n_estimators=n_est, tree_method="hist", enable_categorical=True,
                              eval_metric="auc", n_jobs=N_JOBS, random_state=seed,
                              early_stopping_rounds=ES_ROUNDS if Xv is not None else None,
                              max_cat_to_onehot=8, **p)
        if Xv is not None:
            m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        else:
            m.fit(Xa, ya, verbose=False)
        return m

    params = dict(learning_rate=0.1)
    if tune:
        def obj(t):
            p = dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     max_depth=t.suggest_int("max_depth", 2, 8),
                     min_child_weight=t.suggest_float("min_child_weight", 1, 20, log=True),
                     subsample=t.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                     reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
            m = fit(p, A.iloc[tr], y[tr], A.iloc[va], y[va])
            t.set_user_attr("best_iter", m.best_iteration + 1)
            return roc_auc_score(y[va], m.predict_proba(A.iloc[va])[:, 1])
        params, attrs = _optimize(obj, n_trials, seed)
        best_iter = attrs["best_iter"]
    else:
        best_iter = fit(params, A.iloc[tr], y[tr], A.iloc[va], y[va]).best_iteration + 1
    m = fit(params, A, y, n_est=max(10, int(1.25 * best_iter)))
    return m.predict_proba(B)[:, 1], {**params, "n_estimators": int(1.25 * best_iter)}


def _lgbm(A, y, B, cat_cols, tune, n_trials, seed):
    import lightgbm as lgb
    A, B = _as_category(A, cat_cols), _as_category(B, cat_cols, ref=A)
    tr, va = _inner_split(len(y), y, seed)

    def fit(p, Xa, ya, Xv=None, yv=None, n_est=MAX_TREES):
        m = lgb.LGBMClassifier(n_estimators=n_est, n_jobs=N_JOBS, random_state=seed,
                               verbose=-1, **p)
        cb = [lgb.early_stopping(ES_ROUNDS, verbose=False)] if Xv is not None else None
        m.fit(Xa, ya, eval_set=[(Xv, yv)] if Xv is not None else None,
              eval_metric="auc", callbacks=cb)
        return m

    params = dict(learning_rate=0.1)
    if tune:
        def obj(t):
            p = dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     num_leaves=t.suggest_int("num_leaves", 8, 128, log=True),
                     min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
                     subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                     colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                     reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
            m = fit(p, A.iloc[tr], y[tr], A.iloc[va], y[va])
            t.set_user_attr("best_iter", m.best_iteration_ or MAX_TREES)
            return roc_auc_score(y[va], m.predict_proba(A.iloc[va])[:, 1])
        params, attrs = _optimize(obj, n_trials, seed)
        params["subsample_freq"] = 1
        best_iter = attrs["best_iter"]
    else:
        best_iter = fit(params, A.iloc[tr], y[tr], A.iloc[va], y[va]).best_iteration_ or 100
    m = fit(params, A, y, n_est=max(10, int(1.25 * best_iter)))
    return m.predict_proba(B)[:, 1], {**params, "n_estimators": int(1.25 * best_iter)}


def _cat(A, y, B, cat_cols, tune, n_trials, seed):
    from catboost import CatBoostClassifier

    def to_cb(df):
        df = df.copy()
        for c in cat_cols:
            df[c] = df[c].fillna(-1).astype(int).astype(str)
        return df
    A, B = to_cb(A), to_cb(B)
    tr, va = _inner_split(len(y), y, seed)

    def fit(p, Xa, ya, Xv=None, yv=None, n_est=1500):
        m = CatBoostClassifier(iterations=n_est, eval_metric="AUC", random_seed=seed,
                               thread_count=N_JOBS, verbose=False, cat_features=cat_cols,
                               allow_writing_files=False, **p)
        if Xv is not None:
            m.fit(Xa, ya, eval_set=(Xv, yv), early_stopping_rounds=ES_ROUNDS)
        else:
            m.fit(Xa, ya)
        return m

    params = {}
    if tune:
        def obj(t):
            p = dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     depth=t.suggest_int("depth", 3, 8),
                     l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 10, log=True))
            m = fit(p, A.iloc[tr], y[tr], A.iloc[va], y[va])
            t.set_user_attr("best_iter", m.get_best_iteration() + 1)
            return roc_auc_score(y[va], m.predict_proba(A.iloc[va])[:, 1])
        params, attrs = _optimize(obj, n_trials, seed)
        best_iter = attrs["best_iter"]
    else:
        best_iter = fit(params, A.iloc[tr], y[tr], A.iloc[va], y[va]).get_best_iteration() + 1
    m = fit(params, A, y, n_est=max(10, int(1.25 * best_iter)))
    return m.predict_proba(B)[:, 1], {**params, "iterations": int(1.25 * best_iter)}


# --------------------------------------------------------------------------- TFMs
TABPFN1_CONTEXT = 1000          # design limit of TabPFN v1
TABPFN1_MAX_BAGS = 4
_TABPFN1 = None


def _tabpfn1_model():
    global _TABPFN1
    if _TABPFN1 is None:
        import torch
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        from tabpfn import TabPFNClassifier
        _TABPFN1 = TabPFNClassifier(device="cpu", N_ensemble_configurations=4, seed=0)
    return _TABPFN1


def _tabpfn1(A, y, B, seed, indicators=False):
    """TabPFN v1 on CPU. Context capped at 1000 rows (its design limit): for larger
    training sets we average K stratified context bags (K = min(ceil(n/1000), 4))."""
    # v1 does not handle NaN: median imputation (+ optional missing-value indicators)
    imp = SimpleImputer(strategy="median", add_indicator=indicators).fit(A)
    A, B = imp.transform(A), imp.transform(B)
    clf = _tabpfn1_model()
    rng = np.random.default_rng(seed)
    n = len(y)
    k = 1 if n <= TABPFN1_CONTEXT else min(int(np.ceil(n / TABPFN1_CONTEXT)), TABPFN1_MAX_BAGS)
    preds = []
    for b in range(k):
        if n <= TABPFN1_CONTEXT:
            idx = np.arange(n)
        else:
            idx, _ = train_test_split(np.arange(n), train_size=TABPFN1_CONTEXT,
                                      stratify=y, random_state=int(rng.integers(1 << 31)))
        clf.fit(A[idx], y[idx], overwrite_warning=True)
        p = np.concatenate([clf.predict_proba(B[i:i + 2000])[:, 1]
                            for i in range(0, len(B), 2000)])
        preds.append(p)
    return np.mean(preds, axis=0), {"bags": k, "context": min(n, TABPFN1_CONTEXT)}


def _tabpfn_modern(A, y, B, cat_idx, seed):
    """TabPFN v2/2.5/3 (package `tabpfn` >= 6). Needs GPU + Prior Labs token for v2.5+.
    Set SML_TABPFN_VERSION=v2 to use the Apache-licensed v2 weights."""
    from tabpfn import TabPFNClassifier
    ver = os.environ.get("SML_TABPFN_VERSION", "default")
    if ver == "default":
        clf = TabPFNClassifier(random_state=seed, categorical_features_indices=cat_idx,
                               ignore_pretraining_limits=True)
    else:
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(getattr(ModelVersion, ver.upper()))
        clf.set_params(random_state=seed, categorical_features_indices=cat_idx,
                       ignore_pretraining_limits=True)
    clf.fit(A.to_numpy(float), y)
    return clf.predict_proba(B.to_numpy(float))[:, 1], {"version": ver}


def _tabicl(A, y, B, seed):
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(random_state=seed)
    clf.fit(A, y)
    return clf.predict_proba(B)[:, 1], {}


# --------------------------------------------------------------------------- API
def fit_predict(name, X_tr, y_tr, X_te, num_cols, cat_cols, tune=False, n_trials=10,
                seed=0, sentinels=False):
    family = FAMILY[name]
    t0 = time.perf_counter()
    A, B = _preprocess(family, X_tr, X_te, num_cols, cat_cols,
                       sentinels and family in ("linear", "nn"))
    y = np.asarray(y_tr).astype(int)
    cats = [c for c in cat_cols if c in A.columns]
    if name == "lr":
        p, info = _lr(A, y, B, tune, n_trials, seed)
    elif name == "mlp":
        p, info = _mlp(A, y, B, tune, n_trials, seed)
    elif name == "xgb":
        p, info = _xgb(A, y, B, cats, tune, n_trials, seed)
    elif name == "lgbm":
        p, info = _lgbm(A, y, B, cats, tune, n_trials, seed)
    elif name == "cat":
        p, info = _cat(A, y, B, cats, tune, n_trials, seed)
    elif name == "tabpfn1":
        p, info = _tabpfn1(A, y, B, seed)
    elif name == "tabpfn1_mi":
        p, info = _tabpfn1(A, y, B, seed, indicators=True)
    elif name == "tabpfn":
        p, info = _tabpfn_modern(A, y, B, [A.columns.get_loc(c) for c in cats], seed)
    elif name == "tabicl":
        p, info = _tabicl(A, y, B, seed)
    else:
        raise KeyError(name)
    info["seconds"] = round(time.perf_counter() - t0, 3)
    return np.asarray(p, dtype=float), info


def refit_with_params(name, X_tr, y_tr, X_te, num_cols, cat_cols, params, seed=0,
                      sentinels=False):
    """Fit a model with fixed hyper-parameters (found by HPO on a subsample) on the
    full training fold, without further tuning or early stopping."""
    family = FAMILY[name]
    A, B = _preprocess(family, X_tr, X_te, num_cols, cat_cols,
                       sentinels and family in ("linear", "nn"))
    y = np.asarray(y_tr).astype(int)
    cats = [c for c in cat_cols if c in A.columns]
    p = {k: v for k, v in params.items() if k not in ("seconds", "hpo_subsample")}
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=p["C"], max_iter=2000).fit(A, y)
        return m.predict_proba(B)[:, 1]
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        sizes = {"64": (64,), "128-64": (128, 64), "256-128": (256, 128)}
        m = MLPClassifier(hidden_layer_sizes=sizes[p["h"]], alpha=p["alpha"],
                          learning_rate_init=p["lr"], early_stopping=True,
                          validation_fraction=0.15, n_iter_no_change=15, max_iter=300,
                          batch_size=256, random_state=seed).fit(A, y)
        return m.predict_proba(B)[:, 1]
    if name == "xgb":
        import xgboost as xgb
        A2, B2 = _as_category(A, cats), _as_category(B, cats, ref=A)
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
                              random_state=seed, max_cat_to_onehot=8, **p).fit(A2, y)
        return m.predict_proba(B2)[:, 1]
    if name == "lgbm":
        import lightgbm as lgb
        A2, B2 = _as_category(A, cats), _as_category(B, cats, ref=A)
        m = lgb.LGBMClassifier(n_jobs=N_JOBS, random_state=seed, verbose=-1, **p).fit(A2, y)
        return m.predict_proba(B2)[:, 1]
    if name == "cat":
        from catboost import CatBoostClassifier
        for D in (A, B):
            for c in cats:
                D[c] = D[c].fillna(-1).astype(int).astype(str)
        m = CatBoostClassifier(random_seed=seed, thread_count=N_JOBS, verbose=False,
                               cat_features=cats, allow_writing_files=False, **p).fit(A, y)
        return m.predict_proba(B)[:, 1]
    raise KeyError(name)
