# Phase 1 — Literature Review

**Project:** Do tabular foundation models (TFMs) match or outperform GBDTs for credit default prediction, and under which conditions (sample size, class imbalance, number of features)?

This document has three parts:
1. **Key finding and positioning.**
2. **Annotated bibliography.** What each paper says and how it is used in the project.
3. **Draft "Related Work" section.** Ready to paste into the template and adjust.

---

## 1. Key finding: a nearly identical benchmark already exists

In May 2026, Baesens, Lessmann and ten co-authors posted *Foundation Models for Credit Risk Prediction: A Game Changer?* on arXiv `[baesens2026foundation]`. This is the group behind the reference benchmarks in credit scoring (Baesens 2003, Lessmann 2015, Gunnarsson 2021). The paper:

- Compares 29 methods on 14 PD datasets, including 5 TFMs: TabPFN v1, TabPFNv2, TabPFN-Real, MITRA and TabICL.
- Uses Optuna with up to 20 trials per fold, 5-fold CV, and Friedman and Wilcoxon tests with Holm correction.
- Includes **Taiwan, GMSC, German, HMEQ and Home Credit**, i.e. 4 of our 5 core datasets plus the optional one.
- Finds that TabICL has the best average AUC (0.752 vs. 0.749 for CatBoost). Differences among the top methods are small.
- Finds that the TFM advantage grows as datasets get smaller. This is shown with two analyses: (a) the Spearman correlation between each method's rank and n, and (b) learning curves subsampling from 500 to 15,000 rows.

**Consequence:** if the project only replicated "TFM vs. GBDT on several datasets + a learning curve", it would reproduce an already published result. 


### What that paper leaves open, which becomes our contribution

| Gap in Baesens et al. (2026) | How we address it |
|---|---|
| **Class imbalance:** cited as motivation (low-default portfolios) but **not manipulated**. Their datasets range from 6.7% to 40% default, so the effect of imbalance is confounded with everything else that differs between datasets. | **Controlled π axis:** within each dataset we artificially lower the default rate (natural → 10% → 5% → 2% → 1%) at fixed n. We also cross n × π on Taiwan and GMSC. |
| **Dimensionality:** not manipulated. Moreover, almost all public credit datasets have few features (ours have 10 to 23). | **Controlled m axis:** we add uninformative features (row-permuted copies of real columns, as in Grinsztajn et al. 2022) in proportions of 0×, 1× and 3× d. |
| **Calibration:** in their own Table 4, TabICL wins on AUC but has a **worse Brier score** (0.162) than CatBoost (0.133) and TabPFNv2 (0.132). This is not discussed. In credit, the calibrated PD is what drives provisions (IFRS 9), capital (IRB) and pricing. | We report **discrimination and calibration separately** (AUC/Gini/KS vs. Brier, log-loss and ECE) and analyze how each family's calibration changes as π falls. |
| **Model versions:** they evaluate TabICL v1 and TabPFNv2. TabPFN-2.5, TabPFN-3 and TabICLv2 have been released since. | We include at least one recent generation (TabPFN-3 or 2.5, and TabICL or TabICLv2), with TabPFNv2 as a reproducible anchor because its weights are Apache 2.0. |
| **Between-dataset design:** their analysis of the effect of n is correlational across datasets, except for the learning curve. | **Within-dataset design** (one factor at a time + n × π grid), with replicates per condition: effects are attributable to the manipulated factor. |

**Refined research question, for the proposal and introduction:**
> Given that recent large-scale evidence shows TFMs are competitive with tuned GBDTs for PD modeling on average [baesens2026foundation], *under which controlled data conditions*—training-set size, default rate, and number of uninformative features—does that advantage hold, and does it extend from discrimination to **calibration**?

This question also directly addresses the part of the rubric that asks to explain *"what did you expect, what did you find instead"*. It yields clear, falsifiable hypotheses:

- **H1 (n):** TFMs outperform GBDTs at small n, and the gap closes or reverses at large n. We expect to replicate Baesens et al.
- **H2 (π):** at low default rates, TFMs keep a better AUC than GBDTs because they need no tuning with few positives, but their calibration degrades more. This is open: it is the project's main bet.
- **H3 (m):** GBDTs are robust to uninformative features (Grinsztajn et al. 2022). For TFMs the outcome is uncertain: attention over columns may be diluted.

---

## 2. Annotated bibliography

**Verification status:** 

### A. Tabular foundation models (core of the project)

**[baesens2026foundation]** — *Foundation Models for Credit Risk Prediction: A Game Changer?* (arXiv 2605.18147).
The closest prior work; see Section 1. Used in three places in the report: the introduction (motivation), the methodology (we adopt its statistical protocol, Friedman + Wilcoxon-Holm) and the discussion (comparing our results on n with theirs). A useful detail for our discussion: they report that TabPFNv2 on LGD deteriorates above roughly 8,000 observations and do not explain why.

**[hollmann2023tabpfn]** — TabPFN v1 (ICLR 2023).
Introduces the *Prior-Data Fitted Network*: a transformer pretrained on millions of synthetic datasets that predicts in a single forward pass, with no training. It was limited to about 1,000 rows and 100 features, numeric only. For us, it is the conceptual basis of the method.

**[muller2022transformers]** — *Transformers Can Do Bayesian Inference* (ICLR 2022).
The theoretical foundation: a PFN approximates the **posterior predictive distribution** p(y | x, D) under the prior it was pretrained on. This is the statistical link that fits the course well: in-context learning as amortized Bayesian inference. Used in the methodology section to explain *why* TFMs should win at small n: the prior acts as a regularizer.

**[hollmann2025accurate]** — TabPFNv2 (*Nature*, 2025).
Scales to about 10,000 rows and 500 features, and supports categorical features and missing values. Shows that on small datasets it beats tuned CatBoost and XGBoost, as well as AutoGluon with a 4-hour budget. Useful detail: it includes a SHAP example on the Taiwan dataset itself. Its weights are licensed Apache 2.0 with attribution, making it **the most reproducible TFM for the project**.

**[grinsztajn2025tabpfn25]** — TabPFN-2.5 (arXiv 2511.08667).
Scales to 50,000 rows and 2,000 features (20 times more cells than v2). Reports that a single forward pass beats tuned XGBoost and CatBoost. Its weights carry a non-commercial license, which is fine for academic use.

**[grinsztajn2026tabpfn3]** — TabPFN-3 (arXiv 2605.13986).
The current default of the `tabpfn` package. According to the official README, it supports on the order of 1,000,000 × 200, 100,000 × 2,000 or 1,000 × 20,000 (rows × features) on GPU. On CPU, the package limits training sets to 5,000 rows by default. Its weights are non-commercial and require a login token from Prior Labs. **Practical note:** without a GPU TabPFN is slow; use Colab or a Purdue cluster (e.g. Gilbreth or Anvil, if access is available).

**[qu2025tabicl]** — TabICL (ICML 2025).
Uses a two-stage architecture (column-wise embedding, then row-wise interaction, and finally dataset-level ICL) that scales to 500k samples with modest resources. It is the **best TFM in the credit benchmark** of Baesens et al. Code and weights are open.

**[qu2026tabiclv2]** — TabICLv2 (arXiv 2602.11139).
The improved open version. Whether to use it depends on the available compute.

**[klein2025grounding]** — *Foundation models for tabular data within systemic contexts need grounding*.
A critical caveat: laboratory results for PFNs may overstate how easily they can be used in practice. Useful for the limitations section and for balance.

### B. Trees vs. deep learning on tabular data

**[grinsztajn2022tree]** — *Why do tree-based models still outperform deep learning on typical tabular data?* (NeurIPS 2022).
Identifies three reasons: neural networks are biased toward overly smooth functions, **are not robust to uninformative features**, and are not rotation-invariant. Their manipulation of adding uninformative features is exactly our m axis. It is the direct methodological reference for H3.

**[shwartzziv2022tabular]** — *Tabular data: Deep learning is not all you need* (Information Fusion, 2022).
Shows that XGBoost outperforms several tabular neural networks on those networks' own datasets. The standard reference for the claim that GBDTs are the state of the art.

**[gorishniy2021revisiting]** — FT-Transformer and well-tuned MLP/ResNet (NeurIPS 2021).
Establishes that a well-configured MLP is a strong baseline. Justifies our choice of an MLP as the neural baseline (with FT-Transformer as optional).

**[mcelfresh2023neural]** — *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* (NeurIPS 2023).
Shares the "under which conditions?" spirit of our question. Analyzes which dataset meta-features predict when each family wins. Useful for the discussion.

**[ye2024closer]** — *A Closer Look at Deep Learning on Tabular Data*.
A broad benchmark (basis of TALENT, the toolbox used by Baesens et al.).

**[erickson2025tabarena]** — TabArena (2025).
A living benchmark of 51 datasets. Findings: GBDTs remain strong, deep learning catches up with GBDTs given ensembling and larger budgets, and TFMs stand out on small datasets. It also stresses that **the validation protocol and tuning budget change the rankings**, which justifies giving every model the same HPO budget.

### C. Credit scoring

**[lessmann2015benchmarking]** — The reference benchmark in credit scoring (from memory: 41 classifiers and 8 datasets; to be verified).
Establishes the protocol (multiple metrics, statistical tests across datasets). From memory: ensembles (heterogeneous ensembles and random forest) outperform logistic regression. Confirm the exact figures when reading it.

**[gunnarsson2021deep]** — *Deep learning for credit scoring: Do or don't?*
Concludes that, for tabular credit data, GBM-type models (XGBoost) outperform deep learning; deep learning adds value with unstructured data. This is the *status quo* that TFMs challenge.

**[brown2012experimental]** — Classifiers on imbalanced credit datasets.
Does exactly what we propose on the π axis: **under-samples the "bad" class** to create increasing levels of imbalance. As recalled, it concludes that random forest and gradient boosting cope well with extreme imbalance; confirm when reading it. It is the direct methodological precedent for H2.

**[jiang2023benchmarking]** — Benchmark of imbalance-handling techniques in credit scoring.
Used to justify *not* using SMOTE or resampling techniques in the main benchmark (out of scope) and to mention them as future work.

**[baesens2016credit]** The book *Credit Risk Analytics*, source of the HMEQ dataset.

### D. Evaluation and statistics

**[demsar2006statistical]** — The standard for comparing classifiers over multiple datasets: Friedman test, Nemenyi post-hoc test and critical-difference diagram.

**[garcia2008extension]** — Pairwise comparisons with Holm correction.

**[niculescu2005predicting]** — *Predicting good probabilities with supervised learning*.
Shows that maximum-margin methods such as boosting (in its exponential-loss form) and SVMs push predicted probabilities away from 0 and 1 (a sigmoid-shaped distortion), and that Platt scaling or isotonic regression corrects this. Modern GBDTs trained with log-loss are usually better calibrated, so calibration must be measured rather than assumed. The classic reference for the calibration part, and it motivates an optional extra experiment: does the TFM calibration advantage vanish once GBDTs are recalibrated?

**[akiba2019optuna]** — Optuna, for HPO.

### E. GBDT implementations

**[chen2016xgboost]** **[ke2017lightgbm]**, **[prokhorenkova2018catboost]** — Cite when describing the models.

### F. Datasets

**[yeh2009comparisons]** (Taiwan), **[hofmann1994german]**, **[quinlan1987australian]**, **[kaggle2011gmsc]**, **[baesens2016credit]** (HMEQ) and **[kaggle2018homecredit]** (optional). Details in `02_data_cards.md`.

---

## 3. Draft — *Related Work* (~650 words)

**2. Related Work**

*Credit scoring benchmarks.* Predicting the probability of default (PD) is a binary classification problem on tabular data, and the credit-scoring literature has a long tradition of large comparative studies. Lessmann et al. [lessmann2015benchmarking] benchmarked a large set of classifiers on several retail credit datasets and found that ensemble methods consistently outperform the industry-standard logistic regression. Gunnarsson et al. [gunnarsson2021deep] later showed that deep neural networks do not improve on gradient-boosted trees for tabular credit data, positioning XGBoost-type models as the de facto standard. Class imbalance, which is pervasive in credit portfolios, has been studied by artificially under-sampling defaulters to create increasingly imbalanced versions of the same dataset [brown2012experimental]; a design we adopt.

*Trees versus deep learning on tabular data.* Outside credit risk, several large benchmarks confirm that gradient-boosted decision trees (GBDTs) such as XGBoost [chen2016xgboost], LightGBM [ke2017lightgbm] and CatBoost [prokhorenkova2018catboost] remain hard to beat on medium-sized tabular data [shwartzziv2022tabular, grinsztajn2022tree]. Grinsztajn et al. [grinsztajn2022tree] attribute this partly to the sensitivity of neural networks to uninformative features, which they demonstrate by appending such features to real datasets. Well-tuned MLPs and FT-Transformers are nonetheless strong baselines [gorishniy2021revisiting], and which family wins depends on dataset characteristics [mcelfresh2023neural] and on the validation and tuning protocol [erickson2025tabarena].

*Tabular foundation models.* Prior-data fitted networks (PFNs) are transformers pre-trained on millions of synthetic datasets so that, given a labeled training set as context, they approximate the Bayesian posterior predictive distribution p(y | x, D) in a single forward pass [muller2022transformers]. TabPFN [hollmann2023tabpfn] introduced this idea for small classification problems, and TabPFNv2 [hollmann2025accurate] extended it to mixed-type data with up to 10,000 samples, outperforming tuned GBDTs on small datasets without any hyperparameter optimization. Subsequent work has focused on scale: TabICL [qu2025tabicl] uses a column-then-row architecture to handle hundreds of thousands of samples, and TabPFN-2.5 and TabPFN-3 [grinsztajn2025tabpfn25, grinsztajn2026tabpfn3] raise the supported context size by one to two orders of magnitude. Critics caution that laboratory benchmarks may overstate how easily such models transfer to applied settings [klein2025grounding].

*Foundation models for credit risk.* The closest work to ours is Baesens et al. [baesens2026foundation], who benchmark five tabular foundation models against 24 alternatives on 14 PD datasets and find that TabICL attains the best average AUC, with foundation models relatively stronger on smaller datasets. Their analysis of dataset size relies on rank correlations across heterogeneous datasets and on learning curves, while class imbalance and dimensionality are not manipulated. Moreover, their results suggest a dissociation between discrimination and calibration: the foundation model with the highest AUC has a noticeably worse Brier score than CatBoost. Since calibrated PDs, not just rankings, drive provisioning and capital calculations, this gap matters in practice.

*Our contribution.* We complement this evidence with a controlled, within-dataset design. On five public credit datasets we vary (i) the training-set size, (ii) the default rate, down to low-default regimes of 1–2%, and (iii) the number of uninformative features. For each condition we compare tabular foundation models, three GBDT implementations, an MLP, and a logistic-regression scorecard, evaluating discrimination (AUC, Gini, KS, PR-AUC) and calibration (Brier score, log-loss, expected calibration error) separately. Statistical comparisons follow Demšar [demsar2006statistical].

---

## 4. Open items for this phase

- [ ] Run `python scripts/download_papers.py` locally (arXiv was blocked from the build environment).
- [ ] Obtain the papers through Purdue Libraries: Lessmann 2015, Gunnarsson 2021, Brown & Mues 2012, Jiang 2023, Niculescu-Mizil 2005.
- [ ] Read Baesens et al. 2026 in full (25 pages). This is the priority reading.
- [ ] Verify the entries in the .bib.
- [ ] Send the positioning line to the instructor (optional, but recommended).