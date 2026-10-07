"""
Hand-check helper (Phase 2 and Phase 4 "Done when")
===================================================
Prints every feature for one ticker on one date so you can compare it with
TradingView, or the bars around a trade from a saved trade log.

HOW TO RUN:
  python check_features.py AAPL 2015-06-01            # all features on that date
  python check_features.py --random 5                 # 5 random symbol/dates
  python check_features.py --trades results/trades_top1.csv --n 10
                                                      # bars around 10 random trades
"""

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

parser = argparse.ArgumentParser()
parser.add_argument("symbol", nargs="?")
parser.add_argument("date", nargs="?")
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--trades", type=str, default=None)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()


def load_all():
    parts = [pq.read_table(f).to_pandas() for f in
             [config.PRICES_FILE, config.FEATURES_FILE, config.RANKS_FILE]]
    df = pd.concat(parts, axis=1)
    df["symbol"] = df["symbol"].astype(str)
    return df


def show_row(df, sym, date):
    row = df[(df["symbol"] == sym) & (df["date"] == pd.Timestamp(date))]
    if row.empty:
        print(f"No row for {sym} on {date}")
        return
    print(f"\n=== {sym}  {pd.Timestamp(date).date()}  "
          f"https://www.tradingview.com/chart/?symbol={sym.split('-')[0]} ===")
    print(row.T.iloc[:, 0].to_string())


def show_trade(df, tr):
    s = df[df["symbol"] == tr["symbol"]].reset_index(drop=True)
    entry = s.index[s["date"] == pd.Timestamp(tr["entry_date"])][0]
    exit_ = s.index[s["date"] == pd.Timestamp(tr["exit_date"])][0]
    win = s.iloc[max(0, entry - 3): exit_ + 2][["date", "open", "high", "low", "close", "sma50"]]
    print(f"\n=== {tr['symbol']}  entry {tr['entry_date']} @ {tr['entry']:.2f}  stop {tr['stop']:.2f}  "
          f"exit {tr['exit_date']} @ {tr['exit']:.2f}  {tr['exit_reason']}  {tr['result_R']:+.2f}R ===")
    print(win.to_string(index=False, max_rows=40))


def main():
    df = load_all()
    rng = np.random.default_rng(args.seed)
    if args.trades:
        t = pd.read_csv(args.trades)
        for _, tr in t.sample(min(args.n, len(t)), random_state=args.seed).iterrows():
            show_trade(df, tr)
    elif args.random:
        valid = df[df["rs_rank"].notna()]
        for i in rng.choice(len(valid), args.random, replace=False):
            show_row(df, valid["symbol"].iloc[i], valid["date"].iloc[i])
    else:
        show_row(df, args.symbol, args.date)


if __name__ == "__main__":
    main()
