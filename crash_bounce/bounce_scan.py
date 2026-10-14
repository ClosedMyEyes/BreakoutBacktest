"""
Bounce-short scanner: tomorrow's limit orders
=============================================
The frozen rule from bounce_short.py (prev:touch:1:0, hold 10; holdout +8.08% a trade
on 206 trades, 2010-2026). The crash day: a stock closes 30%+ below its close 3 days
earlier, the first such day in 10 bars, after a 60-day rise of 20%+ before the fall,
passing the backtest's universe ($5+ printed close, top 80% by dollar volume, no price
break in the past year). The next day only, it's shorted with a limit order at the
close the day before the crash day: filled at the open if it opens above that, else at
the limit if the day's high reaches it. No stop; cover at the close of the 10th trading
day after the entry.

Run it each evening after refreshing the data, after spike_short/spike_scan.py:
  python pipeline/build_data.py
  python pipeline/build_features.py
  python spike_short/spike_scan.py
  python crash_bounce/bounce_scan.py
  python crash_bounce/bounce_scan.py --asof 2025-05-07     # what it would have said that evening (not saved)

Prints:
  1. Tomorrow's orders: each stock that crashed today and the price to short it at.
  2. The shorts entered in the last few weeks, with the days left to the cover and
     where each stands.
Both are kept in bounce_signals.csv in the work folder, the forward-test record.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spike_short"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crash_short"))
sys.path.insert(0, HERE)

import config  # noqa: E402
import datastore  # noqa: E402
import engine  # noqa: E402
import engine_short  # noqa: E402
import rules  # noqa: E402
import spike_short as ss  # noqa: E402
from params import add_param_flags, collect  # noqa: E402
import crash_trades as ct  # noqa: E402
import bounce_short as bs  # noqa: E402

LOOKBACK_DAYS = 200              # calendar days of bars loaded: the 60-day rise before the fall, the first-in-10 rule, recent trades
RECENT_BARS = 15                 # trades entered in this many bars are listed
SIGNALS_FILE = os.path.join(config.WORK_DIR, "bounce_signals.csv")
COLUMNS = ["open", "high", "low", "close", "unadj_close", "volume", "sma200", "dv_pct", "in_sp500", "in_sp400",
           "in_sp600", "div", "bad_bar", "bad_bar_dn"]


def orders(panel, ev):
    """Crashes on the last bar: short tomorrow with a limit order at the close before the crash day."""
    level = bs.parse_pick(bs.FROZEN)[0]
    last = panel.date.max()
    out = []
    for r, fall, run60 in zip(ev["row"].to_numpy(), ev["fall"].to_numpy(), ev["run60"].to_numpy()):
        if panel.date[r] != last or panel.last_row[r] != r or ss.junk_bar(panel, r):
            continue                                     # the backtest skips a crash day that's a junk print
        real = panel.col("unadj_close")[r] / panel.c[r]
        line = ct.bounce_level(panel, np.array([r]), level)[0]
        out.append({"symbol": panel.sym_names[panel.sym[r]], "crash_date": pd.Timestamp(panel.date[r]),
                    "fall_3d_%": fall, "rise_60d_%": run60, "close": panel.c[r] * real,
                    "short_limit": line * real, "limit_over_close_%": (line / panel.c[r] - 1) * 100})
    return pd.DataFrame(out)


def recent_trades(panel, ev, borrow_pct):
    """Shorts entered in the last RECENT_BARS bars, through the backtest's own trade
    code. One still running shows as open, its ret_% as if covered at the last close."""
    level, fill, bars, stop = bs.parse_pick(bs.FROZEN)
    args = SimpleNamespace(borrow_pct=borrow_pct, slippage_pct=None, start=None, end=None, stop_slip=0.0)
    t, _ = bs.bounce_trades(panel, ev, level, fill, bars, stop, bs.FROZEN_HOLD, args)
    if t.empty:
        return t
    dates = np.unique(panel.date)
    t = t[t["entry_date"] >= pd.Timestamp(dates[max(len(dates) - RECENT_BARS, 0)])]
    if t.empty:
        return t
    e, x = t["entry_row"].to_numpy(), t["exit_row"].to_numpy()
    unadj = panel.col("unadj_close")
    reason = t["exit_reason"].to_numpy()
    still_open = (reason == "end of data") & (panel.date[x] == dates[-1])
    out = pd.DataFrame({
        "symbol": t["symbol"], "crash_date": t["signal_date"], "entry_date": t["entry_date"],
        "filled": np.where(np.isclose(t["entry_raw"], panel.o[e]), "at the open", "at the limit"),
        "short_px": t["entry_raw"].to_numpy() * unadj[e] / panel.c[e],
        "days_held": t["bars_held"], "days_left": np.where(still_open, bs.FROZEN_HOLD - t["bars_held"], 0),
        "status": np.where(still_open, "open", np.where(reason == "time exit", "covered", reason)),
        "last_px": unadj[x], "ret_%": t["ret_pct"]})
    return out.sort_values("entry_date")


def save_signals(order_list, trades, scanned):
    """Upsert into the forward-test record: one row per (kind, symbol, crash_date).
    False when there was nothing to save."""
    new = pd.concat([order_list.assign(kind="order"), trades.assign(kind="trade")], ignore_index=True)
    if new.empty:
        return False
    new["scanned"] = scanned
    if os.path.exists(SIGNALS_FILE):
        old = pd.read_csv(SIGNALS_FILE)
        for c in ("crash_date", "entry_date", "scanned"):    # a night with only orders has no entry_date
            if c in old:
                old[c] = pd.to_datetime(old[c])
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(["kind", "symbol", "crash_date"], keep="last")
    os.makedirs(os.path.dirname(SIGNALS_FILE) or ".", exist_ok=True)
    new = new.round({c: 4 for c in new.select_dtypes("number").columns})
    new.sort_values(["crash_date", "kind", "symbol"]).to_csv(SIGNALS_FILE, index=False)
    return True


def show(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    return df.round(2).to_string(index=False)


def scan(asof, P, borrow_pct):
    """The panel up to asof's close, tomorrow's orders and the recent trades."""
    cols = COLUMNS + rules.universe_columns(P) + engine_short.exit_columns({"exit": "time"})
    try:
        df = datastore.load_panel(cols, start=asof - pd.Timedelta(days=LOOKBACK_DAYS), end=asof,
                                  universe="all", unlock_holdout=True)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python pipeline/build_data.py, then python pipeline/build_features.py, and try again.")
    if df.empty:
        raise SystemExit(f"No bars in the {LOOKBACK_DAYS} days to {asof.date()}: refresh the data first.")
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    ev = ct.crash_events(panel, rules.universe_mask(panel.df, P).copy())
    return panel, orders(panel, ev), recent_trades(panel, ev, borrow_pct)


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--asof", type=str, default=None, help="Scan as of this date's close (default: latest)")
    parser.add_argument("--borrow-pct", type=float, default=5.0, help="Borrow fee, %% a year, for ret_%%")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write bounce_signals.csv (never written for an --asof scan)")
    args = parser.parse_args()
    P = collect(args, ["universe"])

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(datastore.load_market(["spx_above200"])["date"].max())
    panel, order_list, trades = scan(asof, P, args.borrow_pct)
    last = pd.Timestamp(panel.date.max())

    print("=" * 78)
    print(f"BOUNCE-SHORT SCAN  data through {last.date()}  (rule: a 3-day fall of {ct.FALL:g}%+ after a 60-day "
          f"rise of {ct.RUN_MIN:g}%+; short the next day only, with a limit at the close before the crash day; "
          f"no stop; cover after {bs.FROZEN_HOLD} days)")
    print("=" * 78)
    if last < pd.Timestamp.today().normalize() - pd.Timedelta(days=4) and not args.asof:
        print(f"!! The data ends {last.date()}. Run pipeline/build_data.py and pipeline/build_features.py for today's list.")
    print(f"\nOrders for the next session ({len(order_list)}). Short with a day limit order at short_limit, regular "
          f"hours only: it fills at the open if the stock opens above it. Not filled by the close: the setup is "
          f"gone. No stop. Cover at the close of the {bs.FROZEN_HOLD}th trading day after the fill. "
          f"limit_over_close_% = how far it has to bounce from today's close. Prices are what it actually "
          f"traded at.")
    if order_list.empty:
        print("  no crashes tonight")
    else:
        print(show(order_list.sort_values("fall_3d_%")))
    print(f"\nShorts entered in the last {RECENT_BARS} trading days ({len(trades)}). open = still running (ret_% "
          f"after costs, as if covered at the last close); days_left = trading days to the cover.")
    if trades.empty:
        print("  none")
    else:
        print(show(trades))

    if not (args.no_save or args.asof) and save_signals(order_list, trades, pd.Timestamp.today().normalize()):
        print(f"\nSaved to {SIGNALS_FILE}")       # the record holds only live scans


if __name__ == "__main__":
    main()
