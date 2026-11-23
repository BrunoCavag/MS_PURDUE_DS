#!/bin/bash
# Creates the separate CPU environment used for TabPFN v1 (tabpfn==0.1.10) and applies the
# compatibility patches needed with current PyTorch / scikit-learn.
# Usage:  bash scripts/setup_tabpfn_v1.sh [env_dir]      (default: .venv-tabpfn1)
# Then:   export TFM_PYTHON=<env_dir>/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.venv-tabpfn1}
PY=${PYTHON:-python3}

$PY -m venv "$ENV"
ENV=$(cd "$ENV" && pwd)            # absolute path
"$ENV/bin/python" -m pip install -q --upgrade pip
"$ENV/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/python" -m pip install -q numpy pandas scikit-learn scipy optuna pyyaml requests
"$ENV/bin/python" -m pip install -q --no-deps "tabpfn==0.1.10"

SP=$("$ENV/bin/python" -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('tabpfn').origin))")
echo "tabpfn installed in $SP"

# 1) checkpoint (~100 MB) from the tabpfn_v1 branch of the official repo
mkdir -p "$SP/models_diff"
CK="$SP/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
if [ ! -s "$CK" ]; then
  curl -L --fail -o "$CK" \
    "https://github.com/PriorLabs/TabPFN/raw/tabpfn_v1/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
fi
head -c 2 "$CK" | grep -q PK || { echo "checkpoint download failed (not a zip/torch file)"; exit 1; }

# 2) patches
"$ENV/bin/python" - "$SP" <<'PY'
import sys, pathlib
sp = pathlib.Path(sys.argv[1])
# a) imports removed from torch.nn.modules.transformer in recent PyTorch
p = sp / "layer.py"; s = p.read_text()
old = ("from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, "
       "MultiheadAttention, Linear, Dropout, LayerNorm")
new = ("from typing import Optional\nfrom torch import Tensor\n"
       "from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm\n"
       "from torch.nn.modules.transformer import _get_activation_fn")
if old in s: p.write_text(s.replace(old, new))
# b) scikit-learn >= 1.6 renamed force_all_finite -> ensure_all_finite
for f in sp.rglob("*.py"):
    t = f.read_text()
    if "force_all_finite=" in t: f.write_text(t.replace("force_all_finite=", "ensure_all_finite="))
# c) load the local epoch-42 checkpoint instead of searching for other epochs
p = sp / "scripts" / "transformer_prediction_interface.py"; s = p.read_text()
old = ("            model_file_, model_path_, results_file_ = check_file(e_)\n"
       "            if model_file_ is not None:")
new = ("            model_file_, model_path_, results_file_ = get_file(e_)\n"
       "            if Path(model_path_).is_file():")
if old in s: p.write_text(s.replace(old, new))
print("patches applied")
PY

# 3) smoke test on one German fold
cd src && "$ENV/bin/python" - <<'PY'
import numpy as np
from sml.datasets import load
from sml.splits import load_splits
from sml.models import fit_predict
from sml.metrics import evaluate
ds = load("german"); f = load_splits("german")[0]
tr, te = np.array(f["train"]), np.array(f["test"])
p, _ = fit_predict("tabpfn1", ds.X.iloc[tr], ds.y.iloc[tr], ds.X.iloc[te], ds.num_cols, ds.cat_cols)
print("TabPFN v1 smoke test, German fold 0: AUC =", round(evaluate(ds.y.iloc[te], p)["auc"], 4))
PY
echo "Done. Now run:  export TFM_PYTHON=$ENV/bin/python"
