"""
Phase 2 - data pipeline, end to end.

    python scripts/run_phase2.py

1. verifies raw files against data/raw/CHECKSUMS.md5
2. loads every available dataset through the common schema
3. runs sanity checks (binary target, no leakage columns, feature counts)
4. writes repeated stratified CV splits      -> data/splits/<name>.json
5. writes the Experiment B design table      -> data/processed/experiment_design.csv
6. writes EDA tables and figures             -> results/eda/
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sml import eda  # noqa: E402
from sml.datasets import LOADERS  # noqa: E402
from sml.perturbations import design  # noqa: E402
from sml.splits import make_outer_splits, save_splits  # noqa: E402


def verify_checksums():
    f = ROOT / "data" / "raw" / "CHECKSUMS.md5"
    bad = []
    for line in f.read_text().splitlines():
        h, rel = line.split(maxsplit=1)
        p = ROOT / rel
        if not p.exists():
            bad.append(f"missing  {rel}")
        elif hashlib.md5(p.read_bytes()).hexdigest() != h:
            bad.append(f"mismatch {rel}")
    print("[1] checksums:", "OK" if not bad else bad)


def main():
    verify_checksums()

    datasets = {}
    for name, loader in LOADERS.items():
        try:
            datasets[name] = loader()
        except FileNotFoundError as e:
            print(f"[2] skip {name}: {e}")
    print("[2] loaded:", ", ".join(f"{k} ({v.n}x{v.d})" for k, v in datasets.items()))

    for ds in datasets.values():
        assert ds.y.isin([0, 1]).all()
        assert ds.X.shape[1] == len(ds.num_cols) + len(ds.cat_cols)
        leak = [c for c in ds.X.columns if c.lower() in {"id", "target", "class", "bad"}]
        assert not leak, f"{ds.name}: possible leakage columns {leak}"
    print("[3] sanity checks passed")

    for ds in datasets.values():
        p = save_splits(ds, make_outer_splits(ds))
        print(f"[4] splits -> {p.relative_to(ROOT)}")

    conds = design(datasets)
    out = ROOT / "data" / "processed" / "experiment_design.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([c.as_dict() for c in conds]).to_csv(out, index=False)
    print(f"[5] {len(conds)} feasible conditions -> {out.relative_to(ROOT)}")

    summary = eda.summary_table(datasets)
    eda.fig_landscape(summary)
    eda.fig_missingness(datasets)
    eda.fig_univariate_signal(datasets)
    eda.category_audit(datasets)
    print("[6] EDA written to results/eda/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
