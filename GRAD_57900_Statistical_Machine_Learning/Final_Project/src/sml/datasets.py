"""
Dataset loaders for the credit-default benchmark.

Every loader returns a `CreditDataset` with a common schema:
    X           : pandas DataFrame with raw features (NaN kept as NaN)
    y           : pandas Series, int {0,1}, where 1 = default / bad risk
    num_cols    : list of numeric feature names
    cat_cols    : list of categorical feature names (dtype "category")
    meta        : dict with source, citation key and notes

No imputation, scaling or encoding is done here. All of that is fitted
inside each CV fold (see preprocessing.py) to avoid leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


@dataclass
class CreditDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    num_cols: list[str]
    cat_cols: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def d(self) -> int:
        return self.X.shape[1]

    @property
    def default_rate(self) -> float:
        return float(self.y.mean())


def _finalize(name, df, target, num_cols, cat_cols, meta) -> CreditDataset:
    y = df[target].astype(int).rename("default")
    X = df[num_cols + cat_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in cat_cols:
        X[c] = X[c].astype("string").astype("category")
    assert set(np.unique(y)) <= {0, 1}, f"{name}: target is not binary"
    return CreditDataset(name, X.reset_index(drop=True), y.reset_index(drop=True),
                         list(num_cols), list(cat_cols), meta)


# --------------------------------------------------------------------------- #
# 1. Default of Credit Card Clients (Taiwan) - Yeh & Lien (2009), UCI
# --------------------------------------------------------------------------- #
def load_taiwan() -> CreditDataset:
    df = pd.read_csv(RAW / "taiwan" / "UCI_Credit_Card.csv")
    df = df.drop(columns=["ID"])
    target = "default.payment.next.month"
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    # PAY_x are ordinal repayment-status codes (-2..8): kept numeric on purpose.
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]
    meta = dict(source="UCI ML Repository (id 350); mirror: FederatedAI/FATE on GitHub",
                cite="yeh2009comparisons", unit="credit-card clients, Taiwan 2005",
                target_def="default payment next month (1 = yes)",
                notes="EDUCATION has undocumented codes 0,5,6; MARRIAGE has 0. Kept as-is "
                      "(categorical) and reported in the data card.")
    return _finalize("taiwan", df, target, num_cols, cat_cols, meta)


# --------------------------------------------------------------------------- #
# 2. Statlog German Credit - Hofmann (1994), UCI
# --------------------------------------------------------------------------- #
GERMAN_COLS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
    "class",
]
GERMAN_NUM = ["duration_months", "credit_amount", "installment_rate", "residence_since",
              "age", "existing_credits", "num_dependents"]


def load_german() -> CreditDataset:
    df = pd.read_csv(RAW / "german" / "german.data", sep=r"\s+", header=None,
                     names=GERMAN_COLS)
    df["bad"] = (df["class"] == 2).astype(int)  # 1 = good, 2 = bad in the original
    cat_cols = [c for c in GERMAN_COLS[:-1] if c not in GERMAN_NUM]
    meta = dict(source="UCI ML Repository (Statlog German Credit, id 144); "
                       "mirror: JLZml/Credit-Scoring-Data-Sets on GitHub",
                cite="hofmann1994german", unit="consumer loans, Germany",
                target_def="bad credit risk (original class 2)",
                notes="Original data ships with a 5:1 misclassification cost matrix "
                      "(bad-as-good is 5x worse). Useful for a cost-sensitive check.")
    return _finalize("german", df, "bad", GERMAN_NUM, cat_cols, meta)


# --------------------------------------------------------------------------- #
# 3. Statlog Australian Credit Approval - Quinlan, UCI
# --------------------------------------------------------------------------- #
def load_australian() -> CreditDataset:
    cols = [f"A{i}" for i in range(1, 15)] + ["A15"]
    df = pd.read_csv(RAW / "australian" / "australian.dat", sep=r"\s+", header=None,
                     names=cols)
    # A15: 1 = '+' (approved), 0 = '-' (rejected). We model "bad" = rejected.
    df["bad"] = (df["A15"] == 0).astype(int)
    cat_cols = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]
    num_cols = ["A2", "A3", "A7", "A10", "A13", "A14"]
    meta = dict(source="UCI ML Repository (Statlog Australian Credit Approval, id 143); "
                       "mirror: JLZml/Credit-Scoring-Data-Sets on GitHub",
                cite="quinlan1987australian", unit="credit-card applications, Australia",
                target_def="application rejected (original class '-')",
                notes="CAVEAT: target is an approval decision, not observed default. "
                      "Feature names are anonymised. Missing values were already "
                      "mode/mean-imputed by the donor. Same underlying data as the "
                      "Japanese 'crx' set, so crx is NOT used to avoid duplication.")
    return _finalize("australian", df, "bad", num_cols, cat_cols, meta)


# --------------------------------------------------------------------------- #
# 4. HMEQ (home-equity loans) - Baesens et al. (2016), Credit Risk Analytics
# --------------------------------------------------------------------------- #
def load_hmeq() -> CreditDataset:
    df = pd.read_csv(RAW / "hmeq" / "hmeq.csv")
    cat_cols = ["REASON", "JOB"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["BAD"]]
    meta = dict(source="Baesens, Roesch & Scheule (2016) 'Credit Risk Analytics' companion "
                       "data; mirror: JLZml/Credit-Scoring-Data-Sets on GitHub",
                cite="baesens2016credit", unit="home-equity loans, USA",
                target_def="BAD = 1 if client defaulted or is seriously delinquent",
                notes="Genuine missing values in almost every column (DEBTINC ~21%).")
    return _finalize("hmeq", df, "BAD", num_cols, cat_cols, meta)


# --------------------------------------------------------------------------- #
# 5. Give Me Some Credit (Kaggle, 2011)
# --------------------------------------------------------------------------- #
def load_gmsc() -> CreditDataset:
    df = pd.read_csv(RAW / "gmsc" / "cs-training.csv", index_col=0)
    target = "SeriousDlqin2yrs"
    num_cols = [c for c in df.columns if c != target]
    meta = dict(source="Kaggle competition 'Give Me Some Credit' (cs-training.csv); "
                       "mirror: JLZml/Credit-Scoring-Data-Sets on GitHub",
                cite="kaggle2011gmsc", unit="consumer borrowers, USA",
                target_def="90+ days past due in the next 2 years",
                notes="MonthlyIncome (~20%) and NumberOfDependents (~2.6%) missing. "
                      "Past-due counters contain sentinel codes 96/98 (left raw; "
                      "trees/TFMs can isolate them, linear models get a flag).")
    return _finalize("gmsc", df, target, num_cols, [], meta)


# --------------------------------------------------------------------------- #
# 6. (Optional) Home Credit Default Risk - requires Kaggle API, see scripts/
# --------------------------------------------------------------------------- #
def load_homecredit() -> CreditDataset:
    path = RAW / "homecredit" / "application_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py --kaggle` "
            "after configuring your Kaggle API token and accepting the competition rules.")
    df = pd.read_csv(path).drop(columns=["SK_ID_CURR"])
    target = "TARGET"
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]
    meta = dict(source="Kaggle competition 'Home Credit Default Risk' (main table only)",
                cite="kaggle2018homecredit", unit="consumer cash/revolving loans",
                target_def="TARGET = 1 if client had payment difficulties",
                notes="Only application_train.csv is used (no relational tables).")
    return _finalize("homecredit", df, target, num_cols, cat_cols, meta)


LOADERS = {
    "australian": load_australian,
    "german": load_german,
    "hmeq": load_hmeq,
    "taiwan": load_taiwan,
    "gmsc": load_gmsc,
    "homecredit": load_homecredit,
}
CORE = ["australian", "german", "hmeq", "taiwan", "gmsc"]  # ordered by n


def load(name: str) -> CreditDataset:
    return LOADERS[name]()


def available() -> list[str]:
    out = []
    for k in LOADERS:
        try:
            LOADERS[k]()
            out.append(k)
        except FileNotFoundError:
            pass
    return out
