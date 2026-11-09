# Tabular Foundation Models vs. GBDTs for Credit Default Prediction

Final project, Statistical Machine Learning (Purdue). Submission: **December 17, 2026**.

**Research question.** Do tabular foundation models (TFMs: TabPFN, TabICL) match or beat
gradient-boosted trees (XGBoost, LightGBM, CatBoost) for probability-of-default prediction,
and under which conditions: training-set size *n*, default rate *π*, and number of
uninformative features *m*? We evaluate discrimination **and** calibration.

## Status

| Phase | Content | Status |
|---|---|---|
| 1. Literature | `docs/01_revision_literatura.md`, `docs/references.bib` | draft done; PDFs via script |
| 2. Data pipeline | `src/sml/`, `data/`, `results/eda/`, `docs/02_data_cards.md` | done |
| 3. Models | model registry, HPO, training loop | next |
| 4. Experiments | Exp. A (benchmark), Exp. B (controlled conditions) | pending |
| 5–6. Report | 5–8 pages in course template | pending |

## Layout

```
data/raw/            raw files for 5 public datasets + CHECKSUMS.md5
data/splits/         5x3 repeated stratified CV indices (json.gz), shared by all models
data/processed/      experiment_design.csv (450 controlled conditions)
docs/                literature review, references.bib, data cards
results/eda/         dataset summary, categorical audit, 3 figures
scripts/             run_phase2.py, download_data.py, download_papers.py
src/sml/             datasets, preprocessing, splits, perturbations, eda
papers/              (filled by scripts/download_papers.py)
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_phase2.py          # checksums, splits, design table, EDA
python scripts/download_papers.py     # open-access PDFs into papers/
python scripts/download_data.py       # optional: re-download raw data (add --kaggle for Home Credit)
```

## Datasets

Australian (690), German (1,000), HMEQ (5,960), Taiwan (30,000), Give Me Some Credit (150,000);
optional Home Credit (~307k, via Kaggle API). See `docs/02_data_cards.md` for sources,
target definitions, known issues and licences.
