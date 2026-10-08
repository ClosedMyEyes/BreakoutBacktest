"""
Mode A — Trend Template filter test
===================================
Do stocks passing the template go on to do better than stocks failing it?
No entries or exits: just forward returns from the next day's open to the
close 5 / 20 / 60 trading days later (delisted stocks use their last price).

Statistics are computed by date (plan Phase 5.1 and the review note):
  1. For each day, average across the passing stocks and across the failing ones.
  2. Average those daily values over time.
  3. Judge significance with a t-stat over NON-overlapping days (every h-th
     day for horizon h), so overlapping windows and stocks moving together
     don't inflate it.

Also reports first-day-passing events (stocks newly entering the template),
the big-winner rate (fwdmax60 >= 50%) and breakdowns by decade, regime,
sector and size.

HOW TO RUN:
  python filter_test.py
  python filter_test.py --min-rs 80 --trend-days 63 --max-below-high 15
  python filter_test.py --breakdowns       # print the breakdown tables
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

parser = argparse.ArgumentParser()
add_param_flags(parser, ["universe", "template"])
add_run_flags(parser)
parser.add_argument("--breakdowns", action="store_true", help="Print decade/regime/sector/size tables")
parser.add_argument("--big-winner-pct", type=float, default=50.0)
args = parser.parse_args()

P = collect(args, ["universe", "template"])
H = config.FORWARD_HORIZONS
HMAX = max(H)


def by_date_stats(df, mask, col):
    """Per-day mean of `col` for rows in mask, then the time series."""
    sub = df.loc[mask, ["date", col]].dropna()
    return sub.groupby("date")[col].mean()


def tstat_overlap(series, h):
    """t-stat of the mean of a daily series of h-day forward returns, which
    overlap. Newey-West standard error with h lags, so overlapping days don't
    count as independent. Returns (t, number of non-overlapping h-day periods)."""
    x = series.dropna().to_numpy(np.float64)
    n = len(x)
    n_indep = n // max(1, h)
    if n_indep < 3:
        return None, n_indep
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(h, n - 1) + 1):
        var += 2 * (1 - k / (h + 1)) * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return None, n_indep
    return float(x.mean() / np.sqrt(var / n)), n_indep


def spx_forward(market):
    m = market.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": m["date"]})
    for h in H:
        out[f"spx_fwd{h}"] = m["spx_close"].shift(-h) / m["spx_open"].shift(-1) - 1
    return out


def main():
    t0 = time.time()
    label_cols = [f"fwd{h}" for h in H] + [f"fwdmax{HMAX}"]
    cols = rules.universe_columns(P) + rules.template_columns(P) + label_cols
    if args.breakdowns:
        cols += ["in_sp500", "in_sp400", "in_sp600"]
    df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                              unlock_holdout=args.unlock_holdout, labels=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    uni = rules.universe_mask(df, P)
    tmpl = rules.template_mask(df, P)
    df = df.loc[uni].reset_index(drop=True)
    passed = tmpl[uni]
    df["pass"] = passed

    # First-day-passing: passes today, did not pass on the symbol's previous row
    prev = df.groupby("symbol", observed=True)["pass"].shift(1, fill_value=False).astype(bool)
    df["first_day"] = df["pass"] & ~prev

    big = args.big_winner_pct / 100
    df["big_winner"] = (df[f"fwdmax{HMAX}"] >= big).astype(float).where(df[f"fwdmax{HMAX}"].notna())

    market = pd.read_parquet(config.MARKET_FILE)
    df = df.merge(spx_forward(market), on="date", how="left")

    res = {
        "rows": int(len(df)),
        "pass_rate": round(float(df["pass"].mean() * 100), 2),
        "avg_passing_per_day": round(float(df.groupby("date")["pass"].sum().mean()), 1),
        "first_day_events": int(df["first_day"].sum()),
    }
    p_mask, f_mask, fd_mask = df["pass"].to_numpy(), ~df["pass"].to_numpy(), df["first_day"].to_numpy()
    for h in H:
        col = f"fwd{h}"
        df[f"x{h}"] = df[col] - df[f"spx_fwd{h}"]
        sp = by_date_stats(df, p_mask, col)
        sf = by_date_stats(df, f_mask, col)
        edge = (sp - sf).dropna()
        t, n = tstat_overlap(edge, h)
        res[f"fwd{h}_pass"] = round(float(sp.mean() * 100), 3)
        res[f"fwd{h}_fail"] = round(float(sf.mean() * 100), 3)
        res[f"edge{h}"] = round(float(edge.mean() * 100), 3)
        res[f"t{h}"] = round(t, 2) if t is not None else None
        res[f"n_indep{h}"] = n
        res[f"excess{h}_pass"] = round(float(by_date_stats(df, p_mask, f"x{h}").mean() * 100), 3)
        res[f"median{h}_pass"] = round(float(df.loc[p_mask].groupby("date")[col].median().mean() * 100), 3)
        res[f"pos{h}_pass"] = round(float((df.loc[p_mask, col] > 0).groupby(df.loc[p_mask, "date"]).mean().mean() * 100), 1)
        res[f"fwd{h}_first"] = round(float(by_date_stats(df, fd_mask, col).mean() * 100), 3)
    res["big_winner_pass"] = round(float(by_date_stats(df, p_mask, "big_winner").mean() * 100), 3)
    res["big_winner_fail"] = round(float(by_date_stats(df, f_mask, "big_winner").mean() * 100), 3)

    if args.output_json:
        print("GRID_RESULT:" + json.dumps(res))
        return

    print("=" * 60)
    print(f"MODE A: FILTER TEST  universe={args.universe}  ({time.time() - t0:.1f}s)")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:<22}: {v}")

    if args.breakdowns:
        breakdowns(df, market)


def breakdowns(df, market):
    h = 20 if 20 in H else H[0]
    col = f"fwd{h}"
    symbols = datastore.load_symbols()[["symbol", "sector"]]
    df = df.merge(symbols, on="symbol", how="left")
    df["decade"] = (df["date"].dt.year // 10 * 10).astype(str) + "s"

    mf = datastore.load_market(["spx_above200", "spx_sma50_rising"])
    mf["regime"] = np.where(mf["spx_above200"] == 0, "correction",
                            np.where(mf["spx_sma50_rising"] == 1, "uptrend", "chop"))
    df = df.merge(mf[["date", "regime"]], on="date", how="left")

    size = pd.Series("other", index=df.index)
    for c, name in [("in_sp600", "S&P 600"), ("in_sp400", "S&P 400"), ("in_sp500", "S&P 500")]:
        if c in df:
            size[df[c] > 0] = name
    df["size"] = size

    for key in ["decade", "regime", "sector", "size"]:
        rows = []
        for val, g in df.groupby(key):
            sp = by_date_stats(g, g["pass"].to_numpy(), col)
            sf = by_date_stats(g, ~g["pass"].to_numpy(), col)
            rows.append({key: val, "pass_days": int(g["pass"].sum()),
                         f"fwd{h}_pass": sp.mean() * 100, f"fwd{h}_fail": sf.mean() * 100,
                         "edge": (sp - sf).mean() * 100})
        print(f"\nBy {key} (forward {h}-day return, %, averaged by date):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
