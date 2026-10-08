"""Download the UCI Gallstone dataset (ID 1150) and save a reproducible CSV.

Usage:
    python download_data.py

The original UCI target uses 0=gallstones present and 1=absent. This script
preserves that raw target in data/gallstone_uci.csv. The experiment code recodes
it internally to y=1 for gallstones present, matching the manuscript analysis.
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "gallstone_uci.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def fetch_with_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=1150)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    # UCI returns target separately; put target first to mirror the source workbook.
    target_col = y.columns[0]
    df = pd.concat([y[[target_col]], X], axis=1)
    return df


def main():
    df = fetch_with_ucimlrepo()
    if len(df) != 319:
        raise RuntimeError(f"Expected 319 data rows, got {len(df)}")
    if df.shape[1] != 39:
        raise RuntimeError(f"Expected 39 columns (1 target + 38 inputs), got {df.shape[1]}")
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT}")
    print(f"shape={df.shape}")
    print(df.iloc[:, 0].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
