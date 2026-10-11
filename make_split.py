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


def make_split(symbols, seed=config.SPLIT_SEED):
    """Split by company: a stock that left the exchanges and came back
    (SYMBOL~2) always lands on the same side as SYMBOL."""
    rng = np.random.default_rng(seed)
    symbols = symbols.assign(base=symbols["symbol"].astype(str).str.split("~").str[0])
    bases = symbols.sort_values("symbol").drop_duplicates("base")
    side = {}
    for sector, g in bases.groupby("sector"):
        syms = g["base"].to_numpy().copy()
        rng.shuffle(syms)
        # Alternate assignment after shuffling; a random coin decides which half
        # gets the odd one out so neither half is favoured across sectors
        first = "dev" if rng.random() < 0.5 else "holdout"
        second = "holdout" if first == "dev" else "dev"
        for i, s in enumerate(syms):
            side[s] = first if i % 2 == 0 else second
    out = symbols[["symbol", "sector"]].assign(split=symbols["base"].map(side))
    return out.sort_values("symbol").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing split file (don't, once results have been seen)")
    args = parser.parse_args()
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
