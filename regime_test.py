"""
Regime indicators vs a family's trades
======================================
Brainstorm tool for "is this a 2010s kind of market or a 2020s kind?" (plan
Phase 9). Builds candidate market-wide regime series from the Dev stocks, each
known at the close of day t, and splits a saved trade log by each one's level
on the signal day.

Levels are point-in-time: each day's value is ranked against all earlier days
(expanding percentile, after a year of history), so no future data sets the
cut-offs. low = bottom third so far, high = top third.

Candidates
  dispersion      spread of 6-month returns across tradable stocks (75th - 25th percentile)
  small_vs_large  median 6-month return outside the S&P 500 minus inside it
  template_pct    % of tradable stocks passing the Trend Template
  new_highs_pct   % of tradable stocks closing at a 52-week high
  mom_spread      3-month return of stocks in the top RS decile 3 months ago minus the bottom decile
  new_listings    % of listed stocks that first listed in the past year (IPO appetite)
  spx_vol         S&P 500 21-day realized volatility (annualized %)
  pct_above200    % of stocks above their 200-day
  feedback20      average R of the family's last 20 trades closed before the signal day

Many candidates on one trade log will find something by luck. Pick one with a
reason, then check it in walk-forward windows and the holdout.

HOW TO RUN:
  python setup_trend.py --family f3 --stop-mode structural --f0-seeds 0 --save-trades results/f3.csv
  python regime_test.py --trades results/f3.csv
"""

import argparse
import time

import numpy as np
import pandas as pd

import datastore
import rules

parser = argparse.ArgumentParser()
parser.add_argument("--trades", required=True, help="Trade log from setup_trend.py --save-trades")
parser.add_argument("--min-history", type=int, default=252, help="Days of history before a level counts")
parser.add_argument("--feedback-n", type=int, default=20, help="Closed trades in the feedback average")
args = parser.parse_args()

P = {}   # rule defaults: the same tradable universe and template as the setups
LAG = 63


def daily_regimes():
    cols = (rules.universe_columns(P) + rules.template_columns(P)
            + ["close", "ret126", "hi52_c", "in_sp500", "rs_rank"])
    df = datastore.load_panel(cols, universe="dev")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    uni = rules.universe_mask(df, P)
    tmpl = rules.template_mask(df, P)

    g = df.groupby("symbol", observed=True)
    same = g["date"].shift(LAG).notna()
    ret63 = df["close"] / g["close"].shift(LAG) - 1
    rs_then = g["rs_rank"].shift(LAG)

    u = df.loc[uni, ["date", "ret126", "in_sp500", "close", "hi52_c"]].copy()
    u["tmpl"] = tmpl[uni]
    u["ret63"] = ret63[uni].where(same[uni])
    u["rs_then"] = rs_then[uni]
    by = u.groupby("date")
    q = by["ret126"].quantile([0.25, 0.75]).unstack()
    out = pd.DataFrame({
        "dispersion": (q[0.75] - q[0.25]) * 100,
        "small_vs_large": (u[u["in_sp500"] == 0].groupby("date")["ret126"].median()
                           - u[u["in_sp500"] == 1].groupby("date")["ret126"].median()) * 100,
        "template_pct": by["tmpl"].mean() * 100,
        "new_highs_pct": (u["close"] >= u["hi52_c"]).groupby(u["date"]).mean() * 100,
        "mom_spread": (u[u["rs_then"] >= 90].groupby("date")["ret63"].median()
                       - u[u["rs_then"] <= 10].groupby("date")["ret63"].median()) * 100,
    })

    # New listings: Dev symbols whose first listed day falls in the past year, per listed stock
    syms = datastore.load_symbols()
    dev = datastore.tradable_symbols("dev")
    first = syms.loc[syms["symbol"].isin(dev) & ~syms["symbol"].str.contains("~"), "first_date"]
    first = np.sort(pd.to_datetime(first).to_numpy())
    data_start = df["date"].min() + pd.Timedelta(days=365)   # before this, first_date is just where data begins
    first = first[first > np.datetime64(data_start)]
    listed = df.loc[df["listed"] == 1].groupby("date").size()
    days = listed.index.to_numpy()
    recent = np.searchsorted(first, days, side="right") - np.searchsorted(
        first, days - np.timedelta64(365, "D"), side="right")
    out["new_listings"] = pd.Series(recent / listed.to_numpy() * 100, index=listed.index)
    out.loc[out.index < data_start, "new_listings"] = np.nan

    m = datastore.load_market(["spx_close", "pct_above200"]).set_index("date")
    out["spx_vol"] = m["spx_close"].pct_change().rolling(21).std() * np.sqrt(252) * 100
    out["pct_above200"] = m["pct_above200"]
    return out.sort_index()


def feedback(trades, n):
    """Average R of the last n trades closed before each trade's signal day."""
    ex = trades.sort_values("exit_date")
    ex_dates = ex["exit_date"].to_numpy()
    cum = np.r_[0.0, np.cumsum(ex["result_R"].to_numpy())]
    k = np.searchsorted(ex_dates, trades["signal_date"].to_numpy(), side="left")
    val = (cum[k] - cum[np.maximum(k - n, 0)]) / n
    return np.where(k >= n, val, np.nan)


def main():
    t0 = time.time()
    trades = pd.read_csv(args.trades, parse_dates=["signal_date", "entry_date", "exit_date"])
    trades = trades.sort_values("signal_date").reset_index(drop=True)
    reg = daily_regimes()
    pct = reg.expanding(min_periods=args.min_history).rank(pct=True) * 100

    t = trades[["signal_date", "result_R"]].copy()
    t = t.join(pct.reindex(t["signal_date"]).reset_index(drop=True))
    t["feedback20"] = feedback(trades, args.feedback_n)
    t["feedback20"] = t["feedback20"].expanding(min_periods=50).rank(pct=True) * 100
    names = list(reg.columns) + ["feedback20"]

    print("=" * 72)
    print(f"REGIME TEST  {args.trades}  ({len(t):,} trades, {time.time() - t0:.0f}s)")
    print("=" * 72)
    print("\nTrades split by the indicator's level on the signal day (point-in-time thirds):")
    rows = []
    for c in names:
        lvl = pd.cut(t[c], [-0.1, 100 / 3, 200 / 3, 100.1], labels=["low", "mid", "high"])
        g = t.groupby(lvl, observed=False)["result_R"]
        r = {"indicator": c}
        for k in ["low", "mid", "high"]:
            r[f"{k}_n"] = int(g.size().get(k, 0))
            r[f"{k}_R"] = round(float(g.mean().get(k, np.nan)), 3)
        r["high_minus_low"] = round(r["high_R"] - r["low_R"], 3)
        rows.append(r)
    print(pd.DataFrame(rows).set_index("indicator").to_string())

    print("\nBy year: the family's average R and each indicator's average level (0-100, vs history so far):")
    y = t.groupby(t["signal_date"].dt.year)
    yearly = y[names].mean().round(0)
    yearly.insert(0, "avg_R", y["result_R"].mean().round(2))
    yearly.insert(0, "trades", y.size())
    yearly.index.name = "year"
    print(yearly.to_string())

    print("\nRaw values by year (not percentiles), for reading what each period looked like:")
    raw = reg.groupby(reg.index.year).mean().round(1)
    raw.index.name = "year"
    print(raw.to_string())


if __name__ == "__main__":
    main()
