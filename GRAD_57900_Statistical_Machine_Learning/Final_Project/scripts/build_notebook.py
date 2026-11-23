"""Builds notebooks/proyecto_final_sml.ipynb (the consolidated notebook) and
notebooks/colab_tfm_gpu.ipynb (GPU runs for TabPFN-3 / TabICL)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Tabular Foundation Models vs. Gradient Boosting for Credit Default Prediction
**Statistical Machine Learning — Final Project (Purdue), due December 17, 2026**

**Research question.** Under which controlled data conditions — training-set size *n*, default rate *π*,
number of uninformative features *m* — do tabular foundation models (TFMs) match or beat gradient-boosted
trees (GBDTs) for probability-of-default prediction, and does an advantage in discrimination carry over to
calibration?

This notebook consolidates the whole project:

| § | Phase | What is shown |
|---|---|---|
| 1 | Literature | positioning against the closest prior work, hypotheses H1–H3 |
| 2 | Data pipeline | the five datasets, sanity checks, EDA, CV splits, experimental design |
| 3 | Models | model registry, preprocessing per family, a worked example on one fold |
| 4 | Experiments | Exp. A (benchmark, tuned baselines) and Exp. B (controlled n, π, m) |
| 5 | Statistical analysis | Friedman / Nemenyi / Wilcoxon–Holm, regression of the TFM–GBDT gap |
| 6 | Conclusions & reproducibility | answers to H1–H3, limitations, how to rerun everything |

By default the notebook **reads stored results** (`RUN_EXPERIMENTS = False`). Set it to `True` to rerun
all experiments from scratch (hours on CPU)."""),
code("""import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
RUN_EXPERIMENTS = False          # True = rerun Phase 4 (slow)
FIG = ROOT / "results" / "figures"; TAB = ROOT / "results" / "tables"
def show(name, width=760):
    p = FIG / name
    display(Image(filename=str(p), width=width)) if p.exists() else print("missing:", name)
print("project root:", ROOT)"""),

md("""## 1. Literature and positioning

* **Status quo.** GBDTs dominate tabular credit scoring (Lessmann et al. 2015; Gunnarsson et al. 2021) and
  tabular data in general (Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022).
* **TFMs.** Prior-data fitted networks approximate the Bayesian posterior predictive
  $p(y\\mid x, D)=\\int p(y\\mid x,\\varphi)\\,p(\\varphi\\mid D)\\,d\\varphi$ in one forward pass (Müller et al. 2022):
  TabPFN (2023), TabPFNv2 (*Nature* 2025), TabICL (ICML 2025), TabPFN-2.5 / TabPFN-3 (2025–26).
* **Closest work.** Baesens et al. (2026, arXiv:2605.18147) benchmark 5 TFMs on 14 PD datasets: TabICL has the
  best mean AUC (0.752 vs 0.749 CatBoost) but a worse Brier score (0.162 vs 0.133); size effects are inferred
  across datasets; imbalance and dimensionality are **not manipulated**.
* **Our gap.** A *within-dataset* design that varies n, π and m one at a time, with discrimination and
  calibration reported separately.

**Hypotheses.** H1: TFMs win at small n, the gap closes as n grows. H2: at low π TFMs keep ranking power
but calibration degrades. H3: GBDTs are robust to uninformative features; TFMs may not be.

Full annotated bibliography: `docs/01_literature_review.md`; BibTeX: `docs/references.bib`."""),

md("## 2. Data pipeline\n### 2.1 Datasets (common schema, raw files verified by MD5)"),
code("""from sml.datasets import load, CORE
datasets = {n: load(n) for n in CORE}
pd.DataFrame([dict(dataset=n, n=d.n, d=d.d, num=len(d.num_cols), cat=len(d.cat_cols),
                   default_rate=round(d.default_rate, 4),
                   missing_pct=round(100 * d.X.isna().to_numpy().mean(), 2),
                   target=d.meta["target_def"]) for n, d in datasets.items()])"""),
md("### 2.2 Exploratory analysis"),
code("""summary = pd.read_csv(ROOT / "results/eda/dataset_summary.csv"); display(summary)
for f in ["fig1_dataset_landscape.png", "fig2_missingness.png", "fig3_univariate_signal.png"]:
    display(Image(filename=str(ROOT / "results/eda" / f), width=620))"""),
md("""All five public credit datasets have 10–23 features — public credit data are low-dimensional, which is
why dimensionality is studied by *adding* uninformative features (Exp. B, axis m).

### 2.3 Splits and experimental design"""),
code("""from sml.splits import load_splits
sp = load_splits("german"); print(len(sp), "outer folds stored for German; fold 0:",
      len(sp[0]["train"]), "train /", len(sp[0]["test"]), "test")
design = pd.read_csv(ROOT / "data/processed/experiment_design.csv")
print("Experiment B design:", len(design), "feasible conditions (all reps)")
design.assign(pi=design.pi.fillna("natural")).groupby("dataset").agg(
    n_values=("n_train", lambda s: sorted(s.unique())),
    pi_values=("pi", lambda s: sorted(map(str, s.unique()))),
    m_values=("m_noise", lambda s: sorted(s.unique())))"""),
code("""# Controlled condition example: GMSC, n=1000, default rate 2%, 1x d uninformative features
from sml.perturbations import Condition, make_condition
Xtr, ytr, Xte, yte, nc, cc = make_condition(datasets["gmsc"], Condition("gmsc", 1000, 0.02, 1.0, 2500))
print(Xtr.shape, "train default rate", ytr.mean(), "| test", Xte.shape, yte.mean())
Xtr.filter(like="noise_").head(3)"""),

md("""## 3. Models
| Key | Model | Family / preprocessing | Tuning |
|---|---|---|---|
| `lr` | Logistic regression (L2) | linear: median impute + indicators, scaling, one-hot | Optuna (C) |
| `mlp` | MLP (sklearn, early stopping) | nn: as linear, quantile→normal | Optuna |
| `xgb`, `lgbm`, `cat` | XGBoost, LightGBM, CatBoost | tree: raw, NaN native, categorical codes | Optuna, early stopping |
| `tabpfn1` | TabPFN v1 (CPU) | tfm: raw; ≤1,000-row context, ≤4 bagged contexts | none (in-context) |
| `tabpfn`, `tabicl` | TabPFN-3, TabICL (GPU) | tfm | none |

Worked example on one German fold (default hyper-parameters):"""),
code("""from sml.models import fit_predict, DISPLAY
from sml.metrics import evaluate
ds = datasets["german"]; f = load_splits("german")[0]
tr, te = np.array(f["train"]), np.array(f["test"])
rows = []
for m in ["lr", "xgb", "lgbm", "cat", "mlp"]:
    p, info = fit_predict(m, ds.X.iloc[tr], ds.y.iloc[tr], ds.X.iloc[te], ds.num_cols, ds.cat_cols)
    rows.append(dict(model=DISPLAY[m], **{k: round(v, 4) for k, v in evaluate(ds.y.iloc[te], p).items()
                                          if k in ("auc", "ks", "brier", "ece", "cal_slope")},
                     seconds=info["seconds"]))
pd.DataFrame(rows)"""),

md("## 4. Experiments"),
code("""if RUN_EXPERIMENTS:
    import subprocess
    subprocess.run(["bash", str(ROOT / "scripts/queue_sandbox.sh")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_analysis.py")], check=True)
ra = pd.read_csv(ROOT / "results/exp_a/results.csv")
rb = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results/exp_b").glob("results*.csv"))])
print("Exp A rows:", len(ra), "| Exp B rows:", len(rb))
print("compute (sum of fit+predict seconds):", round(ra.seconds.sum() / 60, 1), "min (A),",
      round(rb.seconds.sum() / 60, 1), "min (B)")"""),
md("""### 4.1 Experiment A — natural conditions, tuned baselines
Every model is scored on exactly the same test rows (for Taiwan/GMSC, a fixed stratified subset on which
TabPFN v1 was evaluated)."""),
code("""for met in ["auc", "brier", "ece"]:
    print(met.upper()); display(pd.read_csv(TAB / f"expA_{met}_table.csv", index_col=0))"""),
code("""show("figA1_benchmark.png"); show("figA2_cd_auc.png", 640); show("figA2b_cd_brier.png", 640)
show("figA3_reliability.png", 640)"""),
code("""S = json.loads((TAB / "summary.json").read_text())
for k in ["friedman_dataset_auc", "friedman_fold_auc", "friedman_fold_brier"]:
    print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in S["expA"][k].items()})
print("\\nWilcoxon-Holm, AUC (reference = best model):")
display(pd.DataFrame(S["expA"]["wilcoxon_auc"]).round(4))
print("Wilcoxon-Holm, Brier:"); display(pd.DataFrame(S["expA"]["wilcoxon_brier"]).round(4))"""),
md("### 4.2 Experiment B — controlled conditions (defaults, no tuning)\n**Axis n (H1)**"),
code("""show("figB_delta_auc.png", 900); show("figB_delta_brier.png", 900)
show("figB_n_auc.png", 900); show("figB_n_brier.png", 900)
display(pd.read_csv(TAB / "expB_delta_auc_n.csv").round(4))"""),
md("**Axis π (H2): discrimination and calibration under low default rates**"),
code("""show("figB_pi_auc.png", 700); show("figB_pi_brier.png", 700)
show("figB_pi_ece.png", 700); show("figB_pi_calslope.png", 700)
display(pd.read_csv(TAB / "expB_delta_auc_pi.csv").round(4))"""),
md("**Axis m (H3): uninformative features**"),
code("""show("figB_m_auc.png", 900)
display(pd.read_csv(TAB / "expB_delta_auc_m.csv").round(4))"""),

md("""### 4.3 Robustness: TabPFN v1 with missing-value indicators
TabPFN v1 cannot read NaN; with median imputation it loses the signal in HMEQ's informative missingness.
Adding indicators recovers most of the gap (Exp. A) and restores the small-n advantage (Exp. B)."""),
code("""display(pd.read_csv(TAB / "expA_robustness_missing_indicators.csv").pivot(index="model", columns="dataset", values="auc").round(4))
b = pd.read_csv(TAB / "expB_all.csv"); h = b[(b.dataset == "hmeq") & b.pi.isna() & (b.m_noise == 0)]
h.groupby(["n_train", "model"]).auc.mean().unstack().round(3)"""),
md("""## 5. Regression of the TFM–GBDT gap
$\\Delta_{c} = \\mathrm{metric}_{TFM,c} - \\overline{\\mathrm{metric}}_{GBDT,c}$ (sign flipped for loss metrics, so
$\\Delta>0$ means the TFM is better), regressed on $\\log_{10} n$, $\\log_{10}\\pi$ and $m$ with dataset fixed
effects; standard errors clustered by dataset×replicate."""),
code("""for met in ["auc", "brier", "ece"]:
    o = S["expB"][f"ols_{met}"]
    print(f"\\nΔ{met.upper()}  (n_obs={o['nobs']}, R²={o['r2']:.3f})")
    display(pd.DataFrame({"coef": o["params"], "se": o["se"], "p": o["p"]}).round(4))
print("Win-rate of TabPFN v1 vs mean / best GBDT, by training size:")
display(pd.read_csv(TAB / "expB_winrate_by_n.csv").round(3))"""),

md("""## 6. Conclusions and reproducibility
| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 sample size | supported for calibration; for AUC once missingness is encoded | Brier/ECE slope on log n p ≤ 0.02; win-rate 63% at n ≤ 500; HMEQ+MI wins at n ≤ 1k, loses at 2.5k |
| H2 imbalance | supported, with a discrimination/calibration split | ΔAUC +0.023 per 10× drop in π (p = 0.002); calibration slope 2.2–2.8 at π = 1% |
| H3 noise features | supported | ΔAUC −0.010 per d noise features (p = 0.007) |

Limitations: TabPFN v1 (≤1,000-row context, CPU), reduced folds/trials on the two large datasets, 2 replicates.
The full discussion is in the report. Reproduce with:

```bash
pip install -r requirements.txt
python scripts/run_phase2.py                 # data checks, splits, design, EDA
bash   scripts/queue_sandbox.sh              # Phase 4 with the CPU budget used here
python scripts/run_analysis.py               # tables + figures
```
Modern TFMs (TabPFN-3, TabICL) on a GPU: `notebooks/colab_tfm_gpu.ipynb` — the same runner with
`--models tabpfn,tabicl`; `run_analysis.py` then includes them automatically."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3",
                                                               "display_name": "Python 3"}})
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "proyecto_final_sml.ipynb")

colab = [
md("""# GPU runs: TabPFN-3 and TabICL (Google Colab)
Runtime → Change runtime type → **GPU**. Upload `proyecto_sml.zip` to the session (or to Drive).
TabPFN ≥ 2.5 weights are non-commercial and need a free Prior Labs token (academic use is allowed).
Set `SML_TABPFN_VERSION=v2` to use the Apache-licensed TabPFNv2 weights instead."""),
code("""!unzip -q -o proyecto_sml.zip -d /content && cd /content/proyecto_sml
%cd /content/proyecto_sml
!pip -q install -r requirements.txt"""),
code("""import os
# os.environ["SML_TABPFN_VERSION"] = "v2"   # optional: Apache-2.0 weights
# TabPFN-3 login (once):  from tabpfn import TabPFNClassifier  -> follow the token prompt
!python scripts/run_experiments.py A --models tabpfn,tabicl --max-test-tfm 3000 --max-folds-large 3 --out results/exp_a
!python scripts/run_experiments.py B --models tabpfn,tabicl --reps 2 --n-test 1000 --min-defaults 30 --axes-only --out results/exp_b_gpu"""),
code("""!cp results/exp_b_gpu/results.csv results/exp_b/results_gpu.csv
!python scripts/run_analysis.py
!zip -qr results_gpu.zip results"""),
md("Download `results_gpu.zip` and replace the local `results/` folder; the consolidated notebook and the "
   "report figures then include TabPFN-3 and TabICL."),
]
nbf.write(nbf.v4.new_notebook(cells=colab, metadata={"accelerator": "GPU"}),
          ROOT / "notebooks" / "colab_tfm_gpu.ipynb")
print("notebooks written")
