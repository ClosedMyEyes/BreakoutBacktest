"""
Spike-short scanner — tomorrow's gap-up shorts
==============================================
The frozen rule from spike_short.py (dev half +9.25% a trade, locked half
+5.45%, 2010-2026): a stock whose close is 100%+ above the lowest close of the
10 bars before it, and that opens the next day at least 2% above that day's
high, is shorted at that open. The stop is 25% above the short, and it's covered
at the close of the 10th trading day after the entry. Same universe as the
backtest: $5+ printed close, top 80% by dollar volume, no price break in the
past year, and only the first gap-up in 10 bars.

Run it each evening after refreshing the data:
  python build_data.py
  python build_features.py
  python spike_scan.py
  python spike_scan.py --asof 2025-05-07     # what it would have said that evening (not saved)

Prints:
  1. The watch list for the next open, with the price the open has to beat.
  2. The setups that triggered in the last few weeks, with the stop, the cover
     day and where each stands (open, stopped, covered).
Both are kept in spike_signals.csv in the work folder, the forward-test record:
the rule can only be judged on trades from here on now that the locked half is used.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config
import datastore
import engine
import engine_short
import rules
import spike_short as ss
from params import add_param_flags, collect

RULE = {"bars": 10, "gain": 100.0, "min_gap": 2.0, "stop": 25.0, "hold": 10}
LOOKBACK_DAYS = 90               # calendar days of bars loaded: the run, the first-in-10 rule, recent trades
RECENT_BARS = 15                 # trades entered in this many bars are listed
SIGNALS_FILE = os.path.join(config.WORK_DIR, "spike_signals.csv")


def raw_gap_days(panel, gain):
    """Bars that opened min_gap% above a qualifying day's high, before the universe
    filter: what spike_candidates' first-in-10 rule counts."""
    sym, o, h = panel.sym, panel.o, panel.h
    cand = np.zeros(len(o), dtype=bool)
    cand[1:] = (gain[:-1] >= RULE["gain"]) & (o[1:] > h[:-1] * (1 + RULE["min_gap"] / 100.0)) \
        & (sym[1:] == sym[:-1])
    return cand


def watch_list(panel, keep, gain):
    """Stocks that qualify at the last bar: short the next open if it's above trigger."""
    last = panel.date.max()
    rows = np.flatnonzero((panel.date == last) & (panel.last_row == np.arange(len(panel.c))))
    gap = raw_gap_days(panel, gain)
    out = []
    for r in rows:
        if not (keep[r] and gain[r] >= RULE["gain"]) or ss.junk_bar(panel, r):
            continue
        lo = max(r - ss.FIRST_IN + 1, 0)
        if gap[lo:r + 1][panel.sym[lo:r + 1] == panel.sym[r]].any():
            continue                                     # already gapped up in the last 10 bars
        real = panel.col("unadj_close")[r] / panel.c[r]
        trigger = panel.h[r] * (1 + RULE["min_gap"] / 100.0)
        out.append({"symbol": panel.sym_names[panel.sym[r]], "signal_date": pd.Timestamp(panel.date[r]),
                    "run_%": gain[r], "close": panel.c[r] * real, "high": panel.h[r] * real,
                    "short_if_open_above": trigger * real, "gap_over_close_%": (trigger / panel.c[r] - 1) * 100})
    return pd.DataFrame(out)


def recent_trades(panel, keep, gain, borrow_pct):
    """Setups that triggered in the last RECENT_BARS bars, through the same short
    engine as the backtest. A trade still running shows as open."""
    cand = ss.spike_candidates(panel, keep, gain, RULE["gain"], "gap", RULE["min_gap"])
    p = ss.exit_params(RULE["stop"], RULE["hold"], borrow_pct)
    t = engine_short.run_signals(panel, cand, p, family="scan")
    if t.empty:
        return t
    dates = np.unique(panel.date)
    t = t[t["entry_date"] >= pd.Timestamp(dates[max(len(dates) - RECENT_BARS, 0)])].copy()
    if t.empty:
        return t
    slip, comm = p["slippage_pct"] / 100.0, p["commission"]
    e, x = t["entry_row"].to_numpy(), t["exit_row"].to_numpy()
    real = panel.col("unadj_close")[e] / panel.c[e]
    still_open = t["exit_reason"].eq("end of data").to_numpy()
    out = pd.DataFrame({
        "symbol": t["symbol"], "signal_date": t["signal_date"], "entry_date": t["entry_date"],
        "short_px": (t["entry"] + comm) / (1 - slip) * real, "stop_px": t["stop"] * real,
        "days_held": t["bars_held"], "days_left": np.where(still_open, RULE["hold"] - t["bars_held"], 0),
        "status": np.where(still_open, "open", t["exit_reason"]),
        "last_px": np.where(still_open, panel.c[x] * real, (t["exit"] - comm) / (1 + slip) * real),
        "ret_%": np.where(still_open, (t["entry"] - panel.c[x]) / t["entry"] * 100, t["ret_pct"])})
    return out.sort_values("entry_date")


def save_signals(watch, trades, scanned):
    """Upsert into the forward-test record: one row per (kind, symbol, signal_date)."""
    new = pd.concat([watch.assign(kind="watch"), trades.assign(kind="trade")], ignore_index=True)
    if new.empty:
        return
    new["scanned"] = scanned
    if os.path.exists(SIGNALS_FILE):
        old = pd.read_csv(SIGNALS_FILE, parse_dates=["signal_date", "entry_date", "scanned"])
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(["kind", "symbol", "signal_date"], keep="last")
    os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
    new.sort_values(["signal_date", "kind", "symbol"]).round(4).to_csv(SIGNALS_FILE, index=False)


def show(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    return df.round(2).to_string(index=False)


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--asof", type=str, default=None, help="Scan as of this date's close (default: latest)")
    parser.add_argument("--borrow-pct", type=float, default=engine_short.EXIT_DEFAULTS["borrow_pct"])
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write spike_signals.csv (never written for an --asof scan)")
    args = parser.parse_args()
    P = collect(args, ["universe"])

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(datastore.load_market(["spx_above200"])["date"].max())
    cols = (["open", "high", "low", "close", "unadj_close", "div", "bad_bar", "bad_bar_dn"]
            + rules.universe_columns(P))
    try:
        df = datastore.load_panel(cols, start=asof - pd.Timedelta(days=LOOKBACK_DAYS), end=asof,
                                  universe="all", unlock_holdout=True)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    if df.empty:
        raise SystemExit(f"No bars in the {LOOKBACK_DAYS} days to {asof.date()}: refresh the data first.")
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    keep = rules.universe_mask(panel.df, P).copy()
    gain = ss.run_gain(panel, RULE["bars"])
    last = pd.Timestamp(panel.date.max())

    watch = watch_list(panel, keep, gain)
    trades = recent_trades(panel, keep, gain, args.borrow_pct)

    print("=" * 78)
    print(f"SPIKE-SHORT SCAN  data through {last.date()}  (rule: 10-day run of {RULE['gain']:g}%+, open "
          f"{RULE['min_gap']:g}%+ above the prior high, stop {RULE['stop']:g}%, cover after {RULE['hold']} days)")
    print("=" * 78)
    if last < pd.Timestamp.today().normalize() - pd.Timedelta(days=4) and not args.asof:
        print(f"!! The data ends {last.date()}. Run build_data.py and build_features.py for today's list.")
    print(f"\nWatch list for the next open ({len(watch)}). Short at the open only if it opens above "
          f"short_if_open_above; stop 25% above your fill; cover at the close of the 10th trading day. "
          f"Prices are what it actually traded at.")
    if watch.empty:
        print("  nothing qualifies tonight")
    else:
        print(show(watch.sort_values("run_%", ascending=False)))
    print(f"\nTriggered in the last {RECENT_BARS} trading days ({len(trades)}). open = still running "
          f"(ret_% at the last close); days_left = trading days to the cover.")
    if trades.empty:
        print("  none")
    else:
        print(show(trades))

    if not (args.no_save or args.asof):            # the record holds only live scans
        save_signals(watch, trades, pd.Timestamp.today().normalize())
        print(f"\nSaved to {SIGNALS_FILE}")


if __name__ == "__main__":
    main()
