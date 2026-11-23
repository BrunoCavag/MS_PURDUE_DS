"""
Phase 4 analysis - produces every table and figure used in the report.

    python scripts/run_analysis.py

Inputs : results/exp_a/{results.csv,preds/}, results/exp_b/results*.csv
Outputs: results/tables/*.csv|json, results/figures/*.png
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sml import analysis as A  # noqa: E402


def main():
    out = {}
    # ------------------------------------------------------------------ Exp A
    if (ROOT / "results/exp_a/preds").exists():
        dfa = A.load_exp_a_paired()
        dfa.to_csv(A.TAB / "expA_paired_scores.csv", index=False) if A.TAB.exists() else None
        A.TAB.mkdir(parents=True, exist_ok=True)
        dfa.to_csv(A.TAB / "expA_paired_scores.csv", index=False)
        A.table_a(dfa)
        for met in ("auc", "brier", "ece"):
            t = A.pretty_table_a(dfa[dfa.model != "tabpfn1_mi"], met)
            t.to_csv(A.TAB / f"expA_{met}_table.csv")
            print(f"\nExp A - {met}\n", t.to_string())
        if "tabpfn1_mi" in dfa.model.unique():          # robustness variant, reported apart
            rob = dfa[dfa.dataset.isin(dfa[dfa.model == "tabpfn1_mi"].dataset.unique())]
            rob.groupby(["dataset", "model"])[["auc", "brier", "ece"]].mean().round(4).to_csv(
                A.TAB / "expA_robustness_missing_indicators.csv")
            dfa = dfa[dfa.model != "tabpfn1_mi"]
        full = dfa.groupby(["dataset", "fold"]).model.nunique()
        kmax = full.max()
        complete = full[full == kmax].index
        dfc = dfa.set_index(["dataset", "fold"]).loc[complete].reset_index()
        out["expA"] = {
            "n_blocks": len(complete), "models": sorted(dfc.model.unique()),
            "friedman_dataset_auc": A.friedman(dfc, "auc", "dataset"),
            "friedman_fold_auc": A.friedman(dfc, "auc", "fold"),
            "friedman_fold_brier": A.friedman(dfc, "brier", "fold"),
            "wilcoxon_auc": A.wilcoxon_holm(dfc, "auc").to_dict("records"),
            "wilcoxon_brier": A.wilcoxon_holm(dfc, "brier").to_dict("records"),
            "wilcoxon_auc_vs_tabpfn1": A.wilcoxon_holm(dfc, "auc", ref="tabpfn1").to_dict("records")
            if "tabpfn1" in dfc.model.unique() else None,
        }
        A.fig_a_dots(dfc)
        A.fig_cd(dfc, "auc", "fold", "figA2_cd_auc.png")
        A.fig_cd(dfc, "brier", "fold", "figA2b_cd_brier.png")
        A.fig_reliability()
        print("\nFriedman (fold, AUC):", out["expA"]["friedman_fold_auc"])

    # ------------------------------------------------------------------ Exp B
    if list((ROOT / "results/exp_b").glob("results*.csv")):
        dfb = A.load_exp_b()
        dfb.to_csv(A.TAB / "expB_all.csv", index=False)
        for axis in ("n", "pi", "m"):
            A.fig_b_axis(dfb, axis, "auc", f"figB_{axis}_auc.png")
            A.fig_b_axis(dfb, axis, "brier", f"figB_{axis}_brier.png")
        A.fig_b_axis(dfb, "pi", "cal_slope", "figB_pi_calslope.png")
        if "tabpfn1" in dfb.model.unique():
            for met in ("auc", "brier", "ece"):
                A.fig_b_delta_panels(dfb, met, "tabpfn1")
        A.fig_b_axis(dfb, "pi", "ece", "figB_pi_ece.png")
        out["expB"] = {"n_rows": len(dfb), "conditions": int(dfb.groupby(
            ["dataset", "n_train", "pi_label", "m_noise", "rep"]).ngroups)}
        if "tabpfn1" in dfb.model.unique():
            for met in ("auc", "brier", "ece"):
                d = A.deltas(dfb, met, "tabpfn1")
                for axis in ("n", "pi", "m"):
                    sl = A.axis_slice(d.assign(model="x"), axis)
                    key = {"n": "n_train", "pi": "pi_label", "m": "m_noise"}[axis]
                    g = sl.groupby(["dataset", key])[["delta_vs_mean", "delta_vs_best"]].mean()
                    g.to_csv(A.TAB / f"expB_delta_{met}_{axis}.csv")
                res, dd = A.regression_b(dfb, met, "tabpfn1")
                out["expB"][f"ols_{met}"] = {
                    "params": res.params.filter(regex="log10|m_noise|Intercept").to_dict(),
                    "se": res.bse.filter(regex="log10|m_noise|Intercept").to_dict(),
                    "p": res.pvalues.filter(regex="log10|m_noise|Intercept").to_dict(),
                    "nobs": int(res.nobs), "r2": float(res.rsquared)}
                (A.TAB / f"expB_ols_{met}.txt").write_text(res.summary().as_text())
            # share of conditions where TFM beats mean / best GBDT, by n bucket
            d = A.deltas(dfb, "auc", "tabpfn1")
            d["n_bucket"] = pd.cut(d.n_train, [0, 600, 1500, 3000, 1e9],
                                   labels=["<=500", "1k", "2.5k", ">=5k"])
            wins = d.groupby("n_bucket", observed=True)[["delta_vs_mean", "delta_vs_best"]].agg(
                lambda s: float((s > 0).mean()))
            wins.to_csv(A.TAB / "expB_winrate_by_n.csv")
            out["expB"]["winrate_by_n"] = wins.to_dict()
            print("\nExp B win-rate of TabPFN v1 by n:\n", wins)
    A.summary_json(out, "summary.json")
    print("\nwritten:", sorted(p.name for p in A.FIG.glob("*.png")))


if __name__ == "__main__":
    main()
