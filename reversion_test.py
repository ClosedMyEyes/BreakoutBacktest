"""
Stretch study — is there short-term mean reversion on the daily chart?
======================================================================
The first test after parking the breakout work (long F3 and short S5 both
stopped working after 2010). No setups, entries or exits: every stock-day in
the universe gets a "stretch" score, stocks are sorted into buckets by that
score each day, and we look at what each bucket did over the next 1 / 3 / 5 /
10 days, against the S&P 500 over the same days.

  Mean reversion = the most stretched-DOWN bucket beats the S&P (buy it) and
  the most stretched-UP bucket lags it (short it).

Stretch measures, all known at the signal day's close:
  move3   3-day close-to-close move in ATR(14) units (the default measure)
  rsi2    RSI over 2 days (simple-average version: up moves / all moves)
  ibs     where the close sits in the day's range, 0 = at the low, 1 = at the high

Forward returns run from the signal day's close (Mark buys at the close;
--entry next_open to start at the next day's open) to the close h days later.
A stock that delists inside the window uses its last close. Windows that
span a price break (either direction) are left out. Costs are NOT charged:
compare the numbers with the round-trip cost printed at the end.

Statistics are by date, as in filter_test.py: average each bucket across
stocks each day, then average those days; t-stats are Newey-West.

HOW TO RUN:
  python reversion_test.py                       # every measure, all years, Dev
  python reversion_test.py --start 2010-01-01    # just the years that broke the breakouts
  python reversion_test.py --measure rsi2        # the breakdown tables for another measure
  python reversion_test.py --entry next_open
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

import config
import datastore
import rules
from params import add_param_flags, add_run_flags, collect
from reports import tstat_overlap

MEASURES = ["move3", "rsi2", "ibs"]
HORIZONS = [1, 3, 5, 10]
MAIN_H = 5
MIN_STOCKS = 50          # skip days with fewer stocks than this in the universe

parser = argparse.ArgumentParser()
add_param_flags(parser, ["universe"])
add_run_flags(parser)
parser.add_argument("--measure", choices=MEASURES, default="move3",
                    help="Measure used for the breakdown tables")
parser.add_argument("--entry", choices=["close", "next_open"], default="close")
args = parser.parse_args()
P = collect(args, ["universe"])


def shift_within(arr, k, codes):
    """arr shifted by k rows (k > 0 = earlier rows), NaN where it would cross a symbol."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = arr[:-k]
        out[k:][codes[k:] != codes[:-k]] = np.nan
    else:
        k = -k
        out[:-k] = arr[k:]
        out[:-k][codes[:-k] != codes[k:]] = np.nan
    return out


def add_measures(df, codes):
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c1, c2, c3 = (shift_within(c, k, codes) for k in (1, 2, 3))
    df["move3"] = (c - c3) / df["atr14"].to_numpy(np.float64)
    d1, d2 = c - c1, c1 - c2
    up = np.maximum(d1, 0) + np.maximum(d2, 0)
    tot = np.abs(d1) + np.abs(d2)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["rsi2"] = np.where(tot > 0, 100 * up / tot, 50.0)
        df["ibs"] = np.where(h > l, (c - l) / (h - l), np.nan)
    df.loc[np.isnan(c3), "rsi2"] = np.nan


def add_forward(df, codes, market):
    """Forward return from the entry price to the close h bars later (or the
    symbol's last close if it ends sooner), minus the S&P over the same days."""
    n = len(df)
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    idx = np.arange(n)
    # last row index of each symbol (rows are sorted by symbol, date)
    last = pd.Series(idx).groupby(codes).transform("max").to_numpy()
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    cb = np.cumsum(brk)

    if args.entry == "close":
        entry = c
        s = 0
    else:
        nxt = np.minimum(idx + 1, last)
        entry = np.where(nxt > idx, o[nxt], np.nan)
        s = 1
    m = market.sort_values("date").reset_index(drop=True)
    spx = pd.DataFrame({"date": m["date"]})
    for hz in HORIZONS:
        j = np.minimum(idx + hz, last)
        ret = c[j] / entry - 1
        ret[j == idx] = np.nan                     # no bars after the signal day
        spans = cb[j] - cb[idx] > 0
        if s:
            spans |= brk[np.minimum(idx + 1, last)]
        ret[spans] = np.nan
        df[f"r{hz}"] = ret.astype(np.float32)
        base = m["spx_close"] if s == 0 else m["spx_open"].shift(-1)
        spx[f"spx{hz}"] = m["spx_close"].shift(-hz) / base - 1
    df = df.merge(spx, on="date", how="left")
    for hz in HORIZONS:
        df[f"x{hz}"] = (df[f"r{hz}"] - df[f"spx{hz}"]).astype(np.float32)
    return df


def add_buckets(df, measure):
    """Per-day percentile of the measure -> bottom 2%, deciles 1-10, top 2%."""
    pct = df.groupby("date")[measure].rank(pct=True)
    dec = np.ceil(pct * 10).clip(1, 10)
    df[f"{measure}_dec"] = dec
    df[f"{measure}_pct"] = pct


BUCKETS = [("bot 2%", lambda p, d: p <= 0.02)] + \
          [(f"D{k}", (lambda k: lambda p, d: d == k)(k)) for k in range(1, 11)] + \
          [("top 2%", lambda p, d: p > 0.98)]


def daily_mean(g, mask, col):
    sub = g.loc[mask, ["date", col]].dropna()
    return sub.groupby("date")[col].mean()


def bucket_table(df, measure, horizons=HORIZONS, keys=None):
    p, d = df[f"{measure}_pct"], df[f"{measure}_dec"]
    rows = []
    for name, fn in BUCKETS:
        if keys and name not in keys:
            continue
        mask = fn(p, d).to_numpy()
        row = {"bucket": name, "avg_" + measure: float(np.nanmean(df.loc[mask, measure]))}
        for hz in horizons:
            s = daily_mean(df, mask, f"x{hz}")
            row[f"vs_spx{hz}"] = s.mean() * 100
        s = daily_mean(df, mask, f"x{MAIN_H}")
        t, _ = tstat_overlap(s, MAIN_H)
        row[f"t{MAIN_H}"] = t
        row[f"raw{MAIN_H}"] = daily_mean(df, mask, f"r{MAIN_H}").mean() * 100
        row[f"up{MAIN_H}_%"] = (df.loc[mask, f"r{MAIN_H}"].dropna() > 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def spread(g, measure):
    """vs-S&P 5-day return of the bottom 2%, D1, D10 and top 2% in group g."""
    p, d = g[f"{measure}_pct"], g[f"{measure}_dec"]
    col = f"x{MAIN_H}"
    out = {"days": g["date"].nunique()}
    for name, mask in [("bot2%", p <= 0.02), ("D1", d == 1), ("D10", d == 10), ("top2%", p > 0.98)]:
        out[name] = daily_mean(g, mask.to_numpy(), col).mean() * 100
    s = daily_mean(g, (d == 1).to_numpy(), col) - daily_mean(g, (d == 10).to_numpy(), col)
    t, _ = tstat_overlap(s.dropna(), MAIN_H)
    out["D1-D10"] = s.mean() * 100
    out["t"] = t
    return out


def breakdowns(df, measure):
    df["block"] = (df["date"].dt.year // 5 * 5)
    df["block"] = df["block"].astype(str) + "-" + (df["block"] + 4).astype(str)
    df["stock_trend"] = np.where(df["close"] >= df["sma200"], "above 200d", "below 200d")
    df.loc[df["sma200"].isna(), "stock_trend"] = "no 200d yet"
    df = df.merge(datastore.load_market(["spx_above200"]), on="date", how="left")
    df["spx"] = np.where(df["spx_above200"] == 1, "S&P above 200d", "S&P below 200d")
    size = pd.Series("other", index=df.index)
    for c, name in [("in_sp600", "S&P 600"), ("in_sp400", "S&P 400"), ("in_sp500", "S&P 500")]:
        size[df[c] > 0] = name
    df["size"] = size
    for key in ["block", "spx", "stock_trend", "size"]:
        rows = [{key: val, **spread(g, measure)} for val, g in df.groupby(key)]
        print(f"\nBy {key}: {MAIN_H}-day return vs S&P (%), measure={measure}  "
              f"(D1-D10 > 0 = reversion)")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    t0 = time.time()
    cols = (rules.universe_columns(P) + ["open", "high", "low", "close", "atr14", "sma200",
                                         "bad_bar", "bad_bar_dn", "in_sp500", "in_sp400", "in_sp600"])
    try:
        df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=20)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy()
    market = pd.read_parquet(config.MARKET_FILE)
    add_measures(df, codes)
    df = add_forward(df, codes, market)

    keep = rules.universe_mask(df, P)
    if args.end is not None:
        keep &= (df["date"] <= pd.Timestamp(args.end)).to_numpy()
    if rules.market_columns(P):
        keep &= rules.market_mask(df["date"], datastore.load_market(rules.market_columns(P)), P)
    df = df.loc[keep].reset_index(drop=True)
    per_day = df.groupby("date")["date"].transform("size")
    df = df.loc[per_day >= MIN_STOCKS].reset_index(drop=True)
    for m in MEASURES:
        add_buckets(df, m)

    tables = {m: bucket_table(df, m) for m in MEASURES}
    if args.output_json:
        res = {"rows": int(len(df))}
        for m, t in tables.items():
            for _, r in t.iterrows():
                if r["bucket"] in ("bot 2%", "D1", "D10", "top 2%"):
                    res[f"{m}_{r['bucket'].replace(' ', '')}_x{MAIN_H}"] = round(float(r[f"vs_spx{MAIN_H}"]), 3)
        print("GRID_RESULT:" + json.dumps(res))
        return

    print("=" * 78)
    print(f"STRETCH STUDY  entry={args.entry}  universe={args.universe}  start={args.start}  "
          f"end={args.end}  market={P['market']}  ({time.time() - t0:.1f}s)")
    print(f"rows {len(df):,}  days {df['date'].nunique():,}  stocks/day "
          f"{len(df) / max(1, df['date'].nunique()):.0f}  "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
    print("=" * 78)
    for m, t in tables.items():
        print(f"\n{m}: buckets by each day's rank (D1 = most stretched down, D10 = most up)")
        print(f"  vs_spxH = average % return over H days minus the S&P's; rawH = without "
              f"subtracting; upH_% = share that rose")
        print(t.round(3).to_string(index=False))
    breakdowns(df, args.measure)

    cost = 2 * config.DEFAULT_SLIPPAGE_PCT
    print(f"\nRound-trip cost to beat: about {cost:.2f}% slippage plus commission "
          f"($0.005/share = 0.05% at a $20 stock, round trip).")
    print("Reversion worth building on = D1 (or bot 2%) clearly above 0 and/or D10 (top 2%) clearly "
          "below 0 on vs_spx, by more than the cost, in the 2010-2014 / 2015-2019 / 2020-2024 rows.")


if __name__ == "__main__":
    main()
