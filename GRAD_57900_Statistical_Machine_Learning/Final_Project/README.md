# Tabular Foundation Models vs. Gradient Boosting for Credit Default Prediction

Final project, Statistical Machine Learning (Purdue). Submission: **December 17, 2026**.

**Research question.** Under which controlled data conditions (training-set size *n*, default rate *π*,
number of uninformative features *m*) do tabular foundation models (TFMs) match or beat gradient-boosted
trees for probability-of-default prediction, and does an advantage in discrimination carry over to calibration?

## Contents

| Path | What |
|---|---|
| `notebooks/proyecto_final_sml.ipynb` | **Consolidated notebook** (all phases, executed, with outputs) |
| `notebooks/colab_tfm_gpu.ipynb` | GPU runs for TabPFN-3 / TabICL (Colab) |
| `docs/01_revision_literatura.md`, `docs/references.bib` | Phase 1: literature review, positioning, BibTeX |
| `docs/02_data_cards.md` | Phase 2: data cards, split protocol, experimental design |
| `src/sml/` | datasets, preprocessing, splits, perturbations, models, metrics, analysis |
| `scripts/` | `run_phase2.py`, `run_experiments.py`, `run_analysis.py`, `queue_sandbox.sh`, downloads |
| `results/exp_a`, `results/exp_b` | raw results (+ test predictions for Exp. A) |
| `results/tables`, `results/figures` | everything used in the report |

## Compute budget actually used (1 CPU, 3 GB RAM, no GPU, no Hugging Face access)

* **TFM**: TabPFN v1 (`tabpfn==0.1.10`, weights from the `tabpfn_v1` branch of PriorLabs/TabPFN on GitHub),
  in a separate env with CPU PyTorch (`TFM_PYTHON`). Two compatibility patches were applied to the old package:
  `layer.py` imports (`Optional`, `Tensor`) and `force_all_finite` → `ensure_all_finite` for scikit-learn ≥ 1.6;
  the checkpoint loader was pointed at the epoch-42 file. Context ≤ 1,000 rows; larger training sets use up
  to 4 stratified context bags whose predictions are averaged.
* **Exp. A**: 5 outer folds for Australian/German/HMEQ, first 3 for Taiwan/GMSC; Optuna 10 trials
  (5 for Taiwan/GMSC), search on ≤ 3,000-row stratified subsample (≤ 5,000 for runs started earlier), refit
  on the full fold. TabPFN v1 scored on a fixed stratified 3,000-row subset of each Taiwan/GMSC test fold;
  the analysis re-scores every model on that subset.
* **Exp. B**: the three one-factor axes (no n × π interaction grid), 2 replicates, defaults + early stopping,
  test ≥ max(1,000, 30/π) rows.

With a GPU, `notebooks/colab_tfm_gpu.ipynb` adds TabPFN-3 and TabICL with the same runner; the full design
(5 × 3 CV, 5 replicates, n × π grid) only needs the default arguments of `run_experiments.py`.

## Datasets

Australian (690), German (1,000), HMEQ (5,960), Taiwan (30,000), Give Me Some Credit (150,000);
optional Home Credit via Kaggle API. Sources, targets and licences: `docs/02_data_cards.md`.