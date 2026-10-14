"""
Compare Norgate's price adjustment types around one date
========================================================
Prints the close a few bars either side of DATE under each adjustment type,
with the day-over-day change, so you can see which one keeps a spin-off or
special dividend from looking like a crash. Needs Norgate Data Updater running.

HOW TO RUN:
  python tools/check_adjust.py ABT 2013-01-02  # AbbVie spin-off
  python tools/check_adjust.py DPZ 2007-05-07  # Domino's special dividend
  python tools/check_adjust.py MSA 2008-01-09 --bars 3
"""

import argparse
import os
import sys

import norgatedata as nd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import config  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("symbol")
parser.add_argument("date")
parser.add_argument("--bars", type=int, default=5, help="Bars either side of the date")
args = parser.parse_args()

TYPES = ["NONE", "CAPITAL", "CAPITALSPECIAL", "TOTALRETURN"]


def main():
    day = pd.Timestamp(args.date)
    start = (day - pd.Timedelta(days=3 * args.bars + 10)).strftime("%Y-%m-%d")
    out = {}
    for t in TYPES:
        df = nd.price_timeseries(args.symbol, start_date=start,
                                 stock_price_adjustment_setting=getattr(nd.StockPriceAdjustmentType, t),
                                 padding_setting=nd.PaddingType.NONE, timeseriesformat="pandas-dataframe")
        c = df[{x.lower(): x for x in df.columns}["close"]]
        out[t] = c
        out[f"{t} chg%"] = (c.pct_change() * 100).round(1)
    table = pd.DataFrame(out)
    table.index = pd.to_datetime(table.index)
    i = table.index.searchsorted(day)
    print(f"{args.symbol} around {day.date()}  (build uses PRICE_ADJUST = {config.PRICE_ADJUST})")
    print(table.iloc[max(0, i - args.bars): i + args.bars + 1].to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
