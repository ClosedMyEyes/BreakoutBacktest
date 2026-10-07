"""
Phase 3 — Dev/Holdout split
===========================
Assigns every symbol, delisted ones included, to Dev or Holdout with a fixed
random seed, balanced so each sector is split in half. Both halves cover the
whole date range, so both live through every market regime (plan 4.3).

The file is written once and then treated as locked: re-running refuses to
overwrite it unless --force is passed. Commit split.csv to the repo.

HOW TO RUN:
  python make_split.py
"""

import argparse
import os

import numpy as np
import pandas as pd

import config

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true",
                    help="Overwrite an existing split file (don't, once results have been seen)")
args = parser.parse_args()


def make_split(symbols, seed=config.SPLIT_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for sector, g in symbols.sort_values("symbol").groupby("sector"):
        syms = g["symbol"].to_numpy().copy()
        rng.shuffle(syms)
        # Alternate assignment after shuffling; a random coin decides which half
        # gets the odd one out so neither half is favoured across sectors
        first = "dev" if rng.random() < 0.5 else "holdout"
        second = "holdout" if first == "dev" else "dev"
        for i, s in enumerate(syms):
            rows.append({"symbol": s, "sector": sector, "split": first if i % 2 == 0 else second})
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def main():
    if os.path.exists(config.SPLIT_FILE) and not args.force:
        print(f"{config.SPLIT_FILE} already exists and is locked. Use --force only if no results "
              f"have been looked at yet.")
        return
    symbols = pd.read_parquet(config.SYMBOLS_FILE)
    split = make_split(symbols)
    split.to_csv(config.SPLIT_FILE, index=False)
    print(f"Saved {config.SPLIT_FILE}")
    print(split.groupby(["sector", "split"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
