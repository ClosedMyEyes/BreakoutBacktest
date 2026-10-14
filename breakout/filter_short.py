"""
Short side, Mode A — do downtrend-template stocks keep falling?
==============================================================
The short-side twin of filter_test.py. No entries or exits: forward returns
from the next day's open to the close 5 / 20 / 60 trading days later, for
stocks passing the downtrend template vs the rest of the short universe.
A short wants passing stocks to do WORSE, so

  short_edge = fail return - pass return     (positive = the template helps a short)

Statistics by date with Newey-West t-stats, as in filter_test.py. Forward
windows that span a downward price break are left out (a drop that can't be
trusted); upward breaks count. Dividends and borrow are not charged here
(setup_short.py charges them).

The market filter defaults to off here so the regime table shows every
regime; --market below200 / downtrend applies it.

HOW TO RUN:
  python breakout/filter_short.py --breakdowns        # with regime / decade / size tables
  python breakout/filter_short.py --market below200
  python breakout/filter_short.py --max-rs 20 --former-leader on
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import config  # noqa: E402
import datastore  # noqa: E402
import rules_short  # noqa: E402
from params import add_param_flags, add_run_flags, collect  # noqa: E402
from reports import by_date_stats, spx_forward, tstat_overlap  # noqa: E402

parser = argparse.ArgumentParser()
add_param_flags(parser, ["s_universe", "s_template"])
add_run_flags(parser)
parser.add_argument("--breakdowns", action="store_true", help="Print regime/decade/size tables")
parser.set_defaults(market="off")
args = parser.parse_args()

P = collect(args, ["s_universe", "s_template"])
H = config.FORWARD_HORIZONS


def main():
    t0 = time.time()
    label_cols = [f"fwd{h}_s" for h in H]
    cols = rules_short.universe_columns(P) + rules_short.template_columns(P) + label_cols
    if args.breakdowns:
        cols += ["in_sp500", "in_sp400", "in_sp600"]
    try:
        df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, labels=True)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python pipeline/build_data.py, then python pipeline/build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    uni = rules_short.universe_mask(df, P)
    if rules_short.market_columns(P):
        uni &= rules_short.market_mask(df["date"], datastore.load_market(rules_short.market_columns(P)), P)
    tmpl = rules_short.template_mask(df, P)
    df = df.loc[uni].reset_index(drop=True)
    df["pass"] = tmpl[uni]
    prev = df.groupby("symbol", observed=True)["pass"].shift(1, fill_value=False).astype(bool)
    df["first_day"] = df["pass"] & ~prev

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
        col = f"fwd{h}_s"
        sp = by_date_stats(df, p_mask, col)
        sf = by_date_stats(df, f_mask, col)
        edge = (sf - sp).dropna()
        t, n = tstat_overlap(edge, h)
        res[f"fwd{h}_pass"] = round(float(sp.mean() * 100), 3)
        res[f"fwd{h}_fail"] = round(float(sf.mean() * 100), 3)
        res[f"short_edge{h}"] = round(float(edge.mean() * 100), 3)
        res[f"t{h}"] = round(t, 2) if t is not None else None
        res[f"n_indep{h}"] = n
        df[f"x{h}"] = df[col] - df[f"spx_fwd{h}"]
        res[f"vs_spx{h}_pass"] = round(float(by_date_stats(df, p_mask, f"x{h}").mean() * 100), 3)
        res[f"down{h}_pass"] = round(float((df.loc[p_mask, col] < 0).groupby(df.loc[p_mask, "date"])
                                           .mean().mean() * 100), 1)
        res[f"fwd{h}_first"] = round(float(by_date_stats(df, fd_mask, col).mean() * 100), 3)

    if args.output_json:
        print("GRID_RESULT:" + json.dumps(res))
        return

    print("=" * 60)
    print(f"SHORT MODE A: DOWNTREND TEMPLATE TEST  market={P['market']}  universe={args.universe}  "
          f"({time.time() - t0:.1f}s)")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:<22}: {v}")
    print("  fwdH = % change over H days (negative = fell); short_edge = fail - pass "
          "(positive = good for shorts); downH_pass = % of passing stocks that fell")
    if args.breakdowns:
        breakdowns(df)


def breakdowns(df):
    h = 20 if 20 in H else H[0]
    col = f"fwd{h}_s"
    df["decade"] = (df["date"].dt.year // 10 * 10).astype(str) + "s"
    mf = datastore.load_market(["spx_above200", "spx_above50", "spx_sma50_rising"])
    mf["regime"] = np.where(mf["spx_above200"] == 1,
                            np.where(mf["spx_sma50_rising"] == 1, "uptrend", "chop (above 200)"),
                            np.where((mf["spx_above50"] == 0) & (mf["spx_sma50_rising"] == 0),
                                     "downtrend", "below 200, other"))
    df = df.merge(mf[["date", "regime"]], on="date", how="left")
    size = pd.Series("other", index=df.index)
    for c, name in [("in_sp600", "S&P 600"), ("in_sp400", "S&P 400"), ("in_sp500", "S&P 500")]:
        if c in df:
            size[df[c] > 0] = name
    df["size"] = size
    for key in ["regime", "decade", "size"]:
        rows = []
        for val, g in df.groupby(key):
            sp = by_date_stats(g, g["pass"].to_numpy(), col)
            sf = by_date_stats(g, ~g["pass"].to_numpy(), col)
            rows.append({key: val, "days": g["date"].nunique(), "pass_rows": int(g["pass"].sum()),
                         f"fwd{h}_pass": sp.mean() * 100, f"fwd{h}_fail": sf.mean() * 100,
                         "short_edge": (sf - sp).mean() * 100})
        print(f"\nBy {key} (forward {h}-day % change, averaged by date):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
