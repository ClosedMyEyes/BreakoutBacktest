"""
Phase 1 — Data build
====================
Pulls every US common stock (active and delisted) plus the market series and
saves them as Parquet files in config.DATA_DIR. Every later step reads these
files, never Norgate directly, so the backtest and the scanner see the same data.

HOW TO RUN:
  python build_data.py                      # full Norgate pull
  python build_data.py --limit 200          # first 200 symbols, for a quick test
  python build_data.py --list-subtypes      # print Norgate security subtypes and exit
  python build_data.py --source synthetic   # fake data, for building/debugging code

Done-when checks (plan Phase 1) are printed at the end: stocks per year,
delisted count, symbols with missing trading days, split spot-check list.
"""

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
from data_sources import NorgateSource, SyntheticSource

parser = argparse.ArgumentParser()
parser.add_argument("--source",  choices=["norgate", "synthetic"], default="norgate")
parser.add_argument("--limit",   type=int, default=0, help="Only the first N symbols (0 = all)")
parser.add_argument("--workers", type=int, default=4, help="Parallel Norgate requests")
parser.add_argument("--list-subtypes", action="store_true",
                    help="Print the security subtype counts and exit (check COMMON_STOCK_SUBTYPES)")
parser.add_argument("--synthetic-symbols", type=int, default=400)
args = parser.parse_args()


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    src = NorgateSource() if args.source == "norgate" else SyntheticSource(args.synthetic_symbols)

    symbols = src.all_symbols()
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"{len(symbols)} symbols listed")

    if args.list_subtypes:
        counts = Counter(src.subtype(s) for s in symbols)
        for k, v in counts.most_common():
            print(f"  {v:7d}  {k}")
        return

    t0 = time.time()
    price_frames, meta_rows, skipped = [], [], Counter()

    def fetch(sym):
        if not src.is_common_stock(sym):
            return sym, None, "not common stock"
        p = src.symbol_prices(sym)
        if p is None or len(p) == 0:
            return sym, None, "no prices"
        return sym, p, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, s) for s in symbols]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                sym, p, why = fut.result()
            except Exception as e:
                skipped[f"error: {type(e).__name__}"] += 1
                continue
            if why:
                skipped[why] += 1
            else:
                price_frames.append(p)
                meta_rows.append(src.symbol_meta(sym, p))
            if i % 500 == 0:
                print(f"  {i}/{len(symbols)}  ({time.time() - t0:.0f}s)")

    prices = pd.concat(price_frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    prices["symbol"] = prices["symbol"].astype("category")
    for c in ["open", "high", "low", "close", "unadj_close", "turnover"]:
        prices[c] = prices[c].astype("float64")
    prices["volume"] = prices["volume"].astype("float64")

    symbols_df = pd.DataFrame(meta_rows).sort_values("symbol").reset_index(drop=True)
    market = src.market()

    prices.to_parquet(config.PRICES_FILE, index=False)
    symbols_df.to_parquet(config.SYMBOLS_FILE, index=False)
    market.to_parquet(config.MARKET_FILE, index=False)
    with open(config.VERSION_FILE, "w") as f:
        json.dump({
            "source": args.source,
            "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "last_date": str(prices["date"].max().date()),
            "symbols": int(len(symbols_df)),
            "rows": int(len(prices)),
        }, f, indent=2)

    print(f"\nSaved {len(prices):,} rows for {len(symbols_df):,} symbols in {time.time() - t0:.0f}s")
    for k, v in skipped.items():
        print(f"  skipped {v} ({k})")
    sanity_checks(prices, symbols_df, market)


def sanity_checks(prices, symbols_df, market):
    sep = "-" * 60
    print(f"\n{sep}\nSANITY CHECKS (plan Phase 1 'Done when')\n{sep}")

    per_year = prices.groupby(prices["date"].dt.year)["symbol"].nunique()
    print("Stocks trading per year:")
    print(per_year.to_string())

    print(f"\nDelisted symbols: {int(symbols_df['delisted'].sum()):,} of {len(symbols_df):,}")

    # Missing trading days: compare each symbol's dates against the market calendar
    cal = pd.DatetimeIndex(sorted(market["date"]))
    gaps = []
    for sym, g in prices.groupby("symbol", observed=True):
        d = pd.DatetimeIndex(g["date"])
        expected = cal[(cal >= d[0]) & (cal <= d[-1])]
        missing = len(expected) - len(d)
        if missing > 5:
            gaps.append((sym, missing))
    print(f"Symbols missing more than 5 trading days inside their life: {len(gaps)}")
    for sym, m in sorted(gaps, key=lambda x: -x[1])[:10]:
        print(f"  {sym}: {m} missing")

    # Split spot-checks: biggest day-over-day jumps in unadjusted/adjusted ratio
    ratio = prices["unadj_close"] / prices["close"]
    jump = ratio / ratio.groupby(prices["symbol"], observed=True).shift(1)
    cand = prices.loc[(jump > 1.4) | (jump < 0.7), ["symbol", "date"]].head(10)
    print("\nSplit days to spot-check on TradingView (adjusted series should be smooth):")
    print(cand.to_string(index=False) if len(cand) else "  none found")

    print("\nAlso check by hand: known 2008-era bankruptcies (e.g. Lehman, Washington Mutual)")
    print("are present with their final prices: look them up in symbols.parquet.")


if __name__ == "__main__":
    main()
