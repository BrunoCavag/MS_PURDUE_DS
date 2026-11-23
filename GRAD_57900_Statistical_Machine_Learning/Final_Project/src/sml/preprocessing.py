"""
Per-model-family preprocessing. Every preprocessor is an sklearn transformer
that must be FIT ON THE TRAINING FOLD ONLY and then applied to the test fold.

Families
--------
linear : logistic regression baseline
         median imputation + missing-indicator, standardisation,
         one-hot categoricals (rare levels grouped), sentinel flags.
tree   : XGBoost / LightGBM / CatBoost
         numeric left raw (NaN handled natively); categoricals ordinal-encoded
         (pass `cat_idx` to the model to use native categorical splits).
nn     : MLP baseline
         median imputation + missing-indicator, quantile->normal transform,
         one-hot categoricals.
tfm    : TabPFN / TabICL
         raw values, NaN kept; categoricals ordinal-encoded. The model docs
         explicitly advise against scaling or one-hot encoding.

The same `family` string is used by the model registry in Phase 3.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, QuantileTransformer,
                                   StandardScaler)

FAMILIES = ("linear", "tree", "nn", "tfm")


class SentinelFlagger(BaseEstimator, TransformerMixin):
    """Adds 0/1 flags for sentinel codes (e.g. GMSC past-due counters 96/98)
    and replaces them by NaN so that downstream imputation handles them."""

    def __init__(self, sentinels=(96, 98), min_count=1):
        self.sentinels = sentinels
        self.min_count = min_count

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.cols_ = [c for c in X.columns
                      if X[c].isin(self.sentinels).sum() >= self.min_count
                      and X[c].dropna().max() in self.sentinels]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cols_:
            mask = X[c].isin(self.sentinels)
            X[f"{c}__sentinel"] = mask.astype(float)
            X.loc[mask, c] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.array(base + [f"{c}__sentinel" for c in self.cols_])


def make_preprocessor(family: str, num_cols: list[str], cat_cols: list[str],
                      handle_sentinels: bool = False) -> ColumnTransformer | Pipeline:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")

    if family in ("tree", "tfm"):
        transformers = [("num", "passthrough", num_cols)]
        if cat_cols:
            transformers.append((
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=np.nan),
                cat_cols))
        ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
        return ct

    # linear / nn
    if family == "linear":
        num_scaler = StandardScaler()
    else:
        num_scaler = QuantileTransformer(output_distribution="normal",
                                         n_quantiles=1000, subsample=100_000,
                                         random_state=0)
    num_steps = []
    if handle_sentinels:
        num_steps.append(("sentinel", SentinelFlagger()))
    num_steps += [("impute", SimpleImputer(strategy="median", add_indicator=True)),
                  ("scale", num_scaler)]
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="constant",
                                               fill_value="__missing__")),
                      ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=0.01,
                                               sparse_output=False))]),
            cat_cols))
    ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def categorical_indices(transformed: pd.DataFrame, cat_cols: list[str]) -> list[int]:
    """Column positions of categoricals after a tree/tfm preprocessor
    (needed by LightGBM/CatBoost/TabPFN `categorical_features_indices`)."""
    return [transformed.columns.get_loc(c) for c in cat_cols if c in transformed.columns]


def _prep_categoricals_for_sklearn(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """SimpleImputer cannot take pandas 'category' with NaN -> cast to object."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def fit_transform_fold(family, ds, train_idx, test_idx, handle_sentinels=None):
    """Convenience: fit preprocessor on train fold, transform both folds.
    Returns (X_tr, X_te, y_tr, y_te, preprocessor)."""
    if handle_sentinels is None:
        handle_sentinels = ds.name == "gmsc" and family in ("linear", "nn")
    X = _prep_categoricals_for_sklearn(ds.X, ds.cat_cols)
    pre = make_preprocessor(family, ds.num_cols, ds.cat_cols, handle_sentinels)
    with warnings.catch_warnings():  # QuantileTransformer n_quantiles > n on tiny folds
        warnings.filterwarnings("ignore", message="n_quantiles")
        X_tr = pre.fit_transform(X.iloc[train_idx])
        X_te = pre.transform(X.iloc[test_idx])
    return X_tr, X_te, ds.y.iloc[train_idx].to_numpy(), ds.y.iloc[test_idx].to_numpy(), pre
