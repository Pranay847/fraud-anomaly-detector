"""Fetch and verify the ULB credit card fraud dataset.

The canonical source is Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud),
which requires an account and an API token. To keep the project reproducible for
anyone who clones it, this script pulls a byte-identical copy from a Hugging Face
mirror, pinned to a specific commit so the file can never change underneath us.

Nothing is trusted blindly: the download is only accepted if the file size, the
SHA-256 digest, the shape, the column names and the fraud count all match the
published dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"

# Mirrors are pinned to a commit SHA, not `main`, so the bytes we verified are
# the bytes a future clone gets.
MIRRORS = [
    "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/90cdcb82de30f04d084599dc37fa054776862e8e/creditcard.csv",
    "https://huggingface.co/datasets/megloughney/creditcardfraud/resolve/a17f3f94350989effe6301b290d3d955c2d52857/creditcard.csv",
    "https://huggingface.co/datasets/TekMonger/creditcard-fraud/resolve/2403b80a76a272921fe41443c49c077e866940d1/creditcard.csv",
]

EXPECTED_BYTES = 150_828_752
# Recorded from the first verified download, after the structural checks below
# confirmed it really is the ULB dataset. Pins the exact bytes from here on.
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, *, strict_hash: bool = True) -> dict:
    """Raise unless `path` is the dataset we expect. Returns a summary dict."""
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(f"size mismatch: got {size:,} bytes, expected {EXPECTED_BYTES:,}")

    digest = sha256_of(path)
    if strict_hash and EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise ValueError(f"sha256 mismatch: got {digest}, expected {EXPECTED_SHA256}")

    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if df.shape != (EXPECTED_ROWS, len(EXPECTED_COLUMNS)):
        raise ValueError(f"shape mismatch: got {df.shape}, expected ({EXPECTED_ROWS}, 31)")
    fraud = int(df["Class"].sum())
    if fraud != EXPECTED_FRAUD:
        raise ValueError(f"fraud count mismatch: got {fraud}, expected {EXPECTED_FRAUD}")
    if df.isna().any().any():
        raise ValueError("unexpected missing values")

    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "fraud": fraud,
        "fraud_rate": fraud / df.shape[0],
    }


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as fh:
            while chunk := response.read(1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="")
    print()


def download(force: bool = False) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists() and not force:
        print(f"{CSV_PATH} already exists - verifying instead of re-downloading.")
        return verify(CSV_PATH)

    errors = []
    for url in MIRRORS:
        # mkstemp hands back an open descriptor; on Windows the file stays
        # locked until it is closed, which breaks the move below.
        fd, tmp_name = tempfile.mkstemp(suffix=".csv", dir=DATA_DIR)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            _download(url, tmp)
            summary = verify(tmp)
        except Exception as exc:  # try the next mirror
            tmp.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
            print(f"  rejected: {exc}")
            continue
        shutil.move(str(tmp), str(CSV_PATH))
        summary["path"] = str(CSV_PATH)
        return summary

    raise RuntimeError(
        "every mirror failed verification:\n  " + "\n  ".join(errors)
        + "\n\nDownload creditcard.csv manually from "
        "https://www.kaggle.com/mlg-ulb/creditcardfraud and place it at "
        f"{CSV_PATH}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    summary = download(force=args.force)
    print("\nDataset verified:")
    print(f"  path       {summary['path']}")
    print(f"  size       {summary['bytes']:,} bytes")
    print(f"  sha256     {summary['sha256']}")
    print(f"  shape      ({summary['rows']:,}, {summary['columns']})")
    print(f"  fraud      {summary['fraud']} ({summary['fraud_rate']:.3%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
