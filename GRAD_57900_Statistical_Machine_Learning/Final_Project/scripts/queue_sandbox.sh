#!/bin/bash
# Job queue actually used in the 1-CPU / no-GPU sandbox (reduced budget; see README).
cd "$(dirname "$0")/.."
mkdir -p logs
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TPY=${TFM_PYTHON:-.venv-tabpfn1/bin/python}       # env with torch + tabpfn==0.1.10
B="B --reps 2 --n-test 1000 --min-defaults 30 --axes-only"
python scripts/run_experiments.py $B --models lr,xgb,lgbm,cat,mlp >> logs/exp_b_classic.log 2>&1
$TPY   scripts/run_experiments.py $B --models tabpfn1             >> logs/exp_b_tabpfn1.log 2>&1
A="A --tune --trials 10 --trials-large 5 --hpo-cap 3000 --max-folds-large 3 --max-test-tfm 3000"
python scripts/run_experiments.py $A --models lr,xgb,lgbm,cat,mlp >> logs/exp_a_classic.log 2>&1
$TPY   scripts/run_experiments.py $A --models tabpfn1             >> logs/exp_a_tabpfn1.log 2>&1
echo ALLDONE > logs/queue_done.flag
# robustness check: TabPFN v1 with missing-value indicators on the datasets with missing data
$TPY scripts/run_experiments.py A --models tabpfn1_mi --datasets hmeq,gmsc --max-folds-large 3 --max-test-tfm 3000 >> logs/exp_a_tabpfn1_mi.log 2>&1
$TPY scripts/run_experiments.py B --models tabpfn1_mi --datasets hmeq --reps 2 --n-test 1000 --min-defaults 30 --axes-only >> logs/exp_b_tabpfn1_mi.log 2>&1
echo ALLDONE2 > logs/queue_done2.flag
