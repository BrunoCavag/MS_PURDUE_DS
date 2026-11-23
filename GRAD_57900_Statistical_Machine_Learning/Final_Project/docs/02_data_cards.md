# Phase 2 — Data Cards

All datasets are public. Raw files are in `data/raw/` with MD5 checksums (`data/raw/CHECKSUMS.md5`) and can be re-downloaded with `python scripts/download_data.py`. They were obtained from public GitHub mirrors of the original files. **The report must cite the official source** ("Official source" field below), not the mirror.

## Summary

| Dataset | n | d (num / cat) | Default rate | neg:pos ratio | % missing cells | Duplicates | TFM regime |
|---|---:|---|---:|---:|---:|---:|---|
| Australian | 690 | 14 (6 / 8) | 55.5%* | 0.8 | 0.0 | 0 | ≤ TabPFNv2 limit |
| German | 1,000 | 20 (7 / 13) | 30.0% | 2.3 | 0.0 | 0 | ≤ TabPFNv2 limit |
| HMEQ | 5,960 | 12 (10 / 2) | 19.9% | 4.0 | 7.4 | 0 | ≤ TabPFNv2 limit |
| Taiwan | 30,000 | 23 (20 / 3) | 22.1% | 3.5 | 0.0 | 35 | ≤ TabPFN-2.5 limit |
| GMSC | 150,000 | 10 (10 / 0) | 6.7% | 14.0 | 2.2 | 609 | > 50k: subsample the context |
| Home Credit (optional) | ~307,000 | ~120 | ~8% | — | — | — | requires Kaggle |

\* In Australian the positive class is "application rejected", not an observed default (see its card).

Duplicates in Taiwan and GMSC are rows identical on every feature and the target. They are plausible in real data (clients with identical profiles), so they are **not removed**; they are reported as a property of the data.

**Note for the report:** the five core datasets have between 10 and 23 features. Public credit datasets are low-dimensional, which is why the dimensionality axis is studied by adding uninformative features in a controlled way (and, time permitting, with Home Credit).

---

## 1. Default of Credit Card Clients (Taiwan)

- **Official source:** UCI Machine Learning Repository, id 350. **Citation:** Yeh & Lien (2009), *Expert Systems with Applications* 36(2).
- **Mirror used:** `FederatedAI/FATE/examples/data/UCI_Credit_Card.csv`, the CSV version identical to the one on Kaggle. The original `.xls` is also included.
- **Unit:** credit-card clients in Taiwan, April to September 2005.
- **Target:** `default.payment.next.month` (1 = defaults on next month's payment).
- **Features:**
  - Credit limit, sex, education, marital status and age.
  - `PAY_0`…`PAY_6`: repayment status over the last 6 months, ordinal codes from −2 to 8. Treated as numeric.
  - `BILL_AMT1–6`: billed amounts.
  - `PAY_AMT1–6`: amounts paid.
- **Known issues:** `EDUCATION` has undocumented codes (0, 5 and 6) and `MARRIAGE` has code 0. They are kept as their own categories to avoid imposing assumptions.
- **License:** CC BY 4.0 according to UCI (verify on the dataset page).

## 2. Statlog German Credit

- **Official source:** UCI, id 144. **Citation:** Hofmann (1994).
- **Mirror used:** `JLZml/Credit-Scoring-Data-Sets` (`german.data` and `german.doc`).
- **Unit:** consumer loans in Germany.
- **Target:** class 2 = bad payer (recoded to 1); class 1 = good payer (0).
- **Features:** 7 numeric (duration, amount, installment rate, years at residence, age, number of credits and number of dependents) and 13 categorical with `Axx` codes (names are in `german.doc`).
- **Useful note:** the original dataset ships with a **5:1 cost matrix**: classifying a bad payer as good costs 5 times more than the opposite error. Useful for an optional cost-sensitive analysis.
- **Ethical caveat:** `personal_status_sex` and `foreign_worker` are protected attributes. They are kept for the benchmark because they are part of the standard dataset, but this deserves a line in the limitations section.
- **License:** CC BY 4.0 according to UCI (verify).

## 3. Statlog Australian Credit Approval

- **Official source:** UCI, id 143 (Quinlan).
- **Mirror used:** `JLZml/Credit-Scoring-Data-Sets`.
- **Unit:** credit-card applications.
- **Target:** `A15` (1 = approved, 0 = rejected). **We model "bad" = rejected**, which is the majority class (55.5%).
- **Features:** 6 numeric and 8 categorical, **anonymized** (A1…A14).
- **Limitations:**
  - The target is an approval decision, not an observed default. This must be stated explicitly in the report. The dataset is included because it is a classic credit-scoring benchmark and the smallest dataset, ideal for the low-n regime.
  - The donor already imputed missing values with the mode or mean, so there are no NaNs.
  - It is the same underlying data as the Japanese `crx` dataset, so **crx is not used**, to avoid duplication.

## 4. HMEQ (Home Equity)

- **Official source:** companion data of Baesens, Roesch & Scheule (2016), *Credit Risk Analytics* (Wiley). Originally from SAS.
- **Mirror used:** `JLZml/Credit-Scoring-Data-Sets`.
- **Unit:** home-equity loans in the US.
- **Target:** `BAD` (1 = defaulted or seriously delinquent).
- **Features:** 10 numeric (loan amount, mortgage due, property value, years at job, derogatory reports, delinquencies, age of oldest credit line, recent inquiries, number of credit lines and debt-to-income ratio) and 2 categorical (`REASON`, `JOB`).
- **Genuine missing values in 11 of 12 columns.** The most affected is `DEBTINC` (~21%). This is the best dataset for seeing how each family handles NaNs natively.

## 5. Give Me Some Credit (GMSC)

- **Official source:** Kaggle competition "Give Me Some Credit" (2011), file `cs-training.csv`. Only the training file is used, because Kaggle's test file has no labels.
- **Mirror used:** `JLZml/Credit-Scoring-Data-Sets`. The original data dictionary (`.xls`) is included.
- **Unit:** consumer borrowers in the US.
- **Target:** `SeriousDlqin2yrs` (90+ days past due within the next 2 years).
- **Features:** 10 numeric: revolving utilization, age, counts of 30–59, 60–89 and 90+ days past due, debt ratio, monthly income, open credit lines, real-estate loans and dependents.
- **Known issues:**
  - Missing values in `MonthlyIncome` (~20%) and `NumberOfDependents` (~2.6%).
  - The three past-due counters contain **sentinel codes 96 and 98**. For the logistic and neural-network families they are converted to NaN plus a flag (`SentinelFlagger`). For trees and foundation models they are left raw, because these models can isolate them with a split.
- **License:** Kaggle competition rules. For submission it is best **not to redistribute** the CSV: the link and the download script suffice. If the instructor requires the files, include them with the source cited.

## 6. Home Credit Default Risk (optional)

- **Official source:** Kaggle competition (2018). Only `application_train.csv`, without the relational tables.
- **Requirements:**
  1. `pip install kaggle`.
  2. API token in `~/.kaggle/kaggle.json`.
  3. Accept the competition rules.
  4. Run `python scripts/download_data.py --kaggle`.
- **Why it is worth it:** it is the only dataset with genuinely high dimensionality (~120 features), which allows contrasting the synthetic m axis with a natural case.

---

## Split protocol (Experiment A: benchmark at natural conditions)

- **Cross-validation:** repeated stratified K-fold, 5 folds × 3 repeats = 15 test folds per dataset. The seed is `20261217`.
- **Stored indices:** in `data/splits/<dataset>.json.gz`. All models use exactly the same partitions, so comparisons are paired, as the Friedman and Wilcoxon tests require.
- **Tuning:** within each training fold, a stratified 20% validation split is held out for HPO and early stopping. The test fold is never touched.
- **Preprocessing:** fitted **on the training fold only**.

| Family | Models | Treatment |
|---|---|---|
| `linear` | Logistic regression | Median + missing indicator, standardization, one-hot (categories below 1% grouped), sentinel flags |
| `tree` | XGBoost, LightGBM, CatBoost | Raw numeric (native NaN), ordinal-encoded categoricals |
| `nn` | MLP | Same as `linear`, but with a quantile-to-normal transform |
| `tfm` | TabPFN, TabICL | Raw values with NaN and ordinal categoricals. The authors advise against scaling or one-hot encoding |

## Experimental design (Experiment B: controlled conditions)

Implemented in `src/sml/perturbations.py`. The full table is in `data/processed/experiment_design.csv` (450 feasible conditions = 90 combinations × 5 replicates).

| Axis | Levels | Datasets |
|---|---|---|
| **n** (training size) | 250, 500, 1k, 2.5k, 5k, 10k, 20k | All (as far as each dataset allows) |
| **π** (default rate in train and test) | natural, 10%, 5%, 2%, 1% | HMEQ, Taiwan, GMSC. Not feasible with enough positives in German and Australian |
| **m** (uninformative features, × d) | 0, 1, 3 | All |
| **n × π** (interaction) | full grid | Taiwan, GMSC |

- **Base point:** n = 2,500 (or whatever the dataset allows), natural π and m = 0. Each axis is varied with the other two fixed at the base point.
- **Sampling:** the test set is drawn first, with a fixed size (5,000 or n/3) and **the same π as the training set**, to simulate a complete low-default portfolio. The training set is then sampled from the remainder.
- **Uninformative features:** row-permuted copies of real columns. They keep the exact marginal distribution (including categorical types and NaNs) but have no relationship with the target.
- **Feasibility:** a condition is dropped if fewer than 10 positives remain in the training set.

> **Runs actually executed (CPU budget):** the three one-factor axes only (no n × π grid), 2 replicates, test sets of at least max(1,000, 30/π) rows — 90 conditions in total. See the README for details.