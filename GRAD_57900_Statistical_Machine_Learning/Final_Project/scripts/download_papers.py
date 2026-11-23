"""
Download the open-access PDFs of the literature review into papers/.

    python scripts/download_papers.py

Paywalled journal articles (EJOR, ESWA) cannot be fetched automatically:
the script lists them at the end with their DOI so you can get them through
the Purdue Libraries proxy. IDs match the keys in docs/references.bib.
"""
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers"

ARXIV = {
    # --- tabular foundation models (core) ---------------------------------
    "P01_baesens2026foundation": "2605.18147",
    "P02_hollmann2023tabpfn": "2207.01848",
    "P04_grinsztajn2025tabpfn25": "2511.08667",
    "P05_grinsztajn2026tabpfn3": "2605.13986",
    "P06_qu2025tabicl": "2502.05564",
    "P07_qu2026tabiclv2": "2602.11139",
    "P08_muller2022transformers": "2112.10510",
    "P22_klein2025grounding": "2505.19825",
    # --- trees vs deep learning on tabular data --------------------------
    "P09_grinsztajn2022tree": "2207.08815",
    "P10_shwartzziv2022tabular": "2106.03253",
    "P11_gorishniy2021revisiting": "2106.11959",
    "P12_mcelfresh2023neural": "2305.02997",
    "P13_ye2024closer": "2407.00956",
    "P26_erickson2025tabarena": "2506.16791",
    # --- GBDT implementations --------------------------------------------
    "P18_chen2016xgboost": "1603.02754",
    "P20_prokhorenkova2018catboost": "1706.09516",
}
OPEN = {
    "P03_hollmann2025nature": "https://www.nature.com/articles/s41586-024-08328-6.pdf",
    "P17_demsar2006statistical": "https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf",
    "P25_garcia2008extension": "https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf",
    "P19_ke2017lightgbm": "https://proceedings.neurips.cc/paper/2017/file/"
                          "6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
}
PAYWALLED = {
    "P14_lessmann2015benchmarking": "10.1016/j.ejor.2015.05.030",
    "P15_gunnarsson2021deep": "EJOR 295(1):292-305 (2021) - search title in Purdue Libraries",
    "P16_brown2012experimental": "Expert Syst. Appl. 39(3):3446-3453 (2012)",
    "P21_yeh2009comparisons": "Expert Syst. Appl. 36(2):2473-2480 (2009)",
    "P24_jiang2023benchmarking": "10.1016/j.eswa.2022.118878",
    "P23_niculescu2005predicting": "ICML 2005 - 'Predicting good probabilities with "
                                   "supervised learning'",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (academic use; STAT final project)"}


def get(url, dst):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    dst.write_bytes(data)
    return len(data)


def main():
    OUT.mkdir(exist_ok=True)
    jobs = {k: f"https://arxiv.org/pdf/{v}" for k, v in ARXIV.items()} | OPEN
    failed = []
    for key, url in jobs.items():
        dst = OUT / f"{key}.pdf"
        if dst.exists():
            print(f"skip  {dst.name}")
            continue
        try:
            size = get(url, dst)
            print(f"ok    {dst.name} ({size / 1e6:.1f} MB)")
            time.sleep(3)  # be polite to arXiv
        except Exception as e:  # noqa: BLE001
            failed.append((key, url, e))
            print(f"FAIL  {key}: {e}")
    print("\nGet these through Purdue Libraries (paywalled):")
    for k, ref in PAYWALLED.items():
        print(f"  {k}: {ref}")
    if failed:
        print("\nFailed automatic downloads (retry or open manually):")
        for k, u, e in failed:
            print(f"  {k}: {u}")


if __name__ == "__main__":
    main()
