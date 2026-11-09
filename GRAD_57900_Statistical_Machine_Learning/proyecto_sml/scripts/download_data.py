"""
Re-download the raw data from public sources and verify checksums.

    python scripts/download_data.py            # 5 core datasets (no login needed)
    python scripts/download_data.py --kaggle   # + Home Credit (needs Kaggle API token)

The raw files are already shipped in data/raw/, so this script is only needed
to prove provenance or to rebuild from scratch. Core files come from public
GitHub mirrors of the original UCI / Kaggle / book files; the official pages
are listed in docs/02_data_cards.md and should be cited in the report.
"""
import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

JLZ = "https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master"
FILES = {
    "taiwan/UCI_Credit_Card.csv":
        "https://raw.githubusercontent.com/FederatedAI/FATE/master/examples/data/UCI_Credit_Card.csv",
    "taiwan/taiwan_original.xls":
        f"{JLZ}/1.%20UCI%20Repository/Taiwan/default%20of%20credit%20card%20clients.xls",
    "german/german.data": f"{JLZ}/1.%20UCI%20Repository/German/german.data",
    "german/german.doc": f"{JLZ}/1.%20UCI%20Repository/German/german.doc",
    "australian/australian.dat": f"{JLZ}/1.%20UCI%20Repository/Australian/australian.dat",
    "australian/australian.doc": f"{JLZ}/1.%20UCI%20Repository/Australian/australian.doc",
    "hmeq/hmeq.csv": f"{JLZ}/6.%20Credit%20risk%20analysis/hmeq/hmeq.csv",
    "gmsc/cs-training.csv": f"{JLZ}/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv",
    "gmsc/gmsc_data_dictionary.xls":
        f"{JLZ}/3.%20Kaggle/Give%20Me%20Some%20Credit/Data%20Dictionary.xls",
}


def expected_md5():
    out = {}
    f = RAW / "CHECKSUMS.md5"
    if f.exists():
        for line in f.read_text().splitlines():
            h, rel = line.split(maxsplit=1)
            out[rel.replace("data/raw/", "")] = h
    return out


def fetch_core(force=False):
    md5 = expected_md5()
    for rel, url in FILES.items():
        dst = RAW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not force:
            status = "present"
        else:
            urllib.request.urlretrieve(url, dst)
            status = "downloaded"
        h = hashlib.md5(dst.read_bytes()).hexdigest()
        ok = md5.get(rel) in (None, h)
        print(f"{'OK ' if ok else 'BAD'} {status:10s} {rel}")
        if not ok:
            sys.exit(f"checksum mismatch for {rel}")


def fetch_homecredit():
    """Requires: pip install kaggle; ~/.kaggle/kaggle.json; and accepting the
    competition rules at https://www.kaggle.com/c/home-credit-default-risk/rules"""
    dst = RAW / "homecredit"
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", "home-credit-default-risk",
                    "-f", "application_train.csv", "-p", str(dst)], check=True)
    for z in dst.glob("*.zip"):
        subprocess.run(["unzip", "-o", str(z), "-d", str(dst)], check=True)
        z.unlink()
    print("Home Credit ready:", list(dst.iterdir()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--kaggle", action="store_true", help="also fetch Home Credit")
    a = ap.parse_args()
    fetch_core(a.force)
    if a.kaggle:
        fetch_homecredit()
