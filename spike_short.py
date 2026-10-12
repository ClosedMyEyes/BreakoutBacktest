"""
Spike shorts — real trades with stops
=====================================
reversion_test.py found that stocks that just spiked fade: shorting the close of
the first day a stock is 50%+ up in 3 days, or (a small-cap day trader's rule)
the open of the first gap-up day after a big run, made money in most years
since 2010. Its tables had no stop, and about 1 trade in 20 kept squeezing
another 35-50%. This script trades the setups through the short engine
(engine_short.simulate_short) with stops, so the squeezes are capped and every
cost is charged: slippage, commission, borrow and dividends.

Setups (each the first of its kind in 10 bars of a stock):
  run      the close is X%+ above the lowest close of the K bars before it
           (K = 3: a spike; K = 10: a run)
  close    short the close of the first day the run qualifies
  gap      the run qualified at yesterday's close and today opens above
           yesterday's high: short today's open (the gap-up blow-off)

Exit: a stop S% above the short price (filled at the open if the stock gaps
through it; on the gap entry day any high at the stop counts), else cover at the
close `--hold` bars after the entry. A grid of K = 3 / 10, X = 50 / 100%,
close / gap and stops none / 15 / 25 / 40%.

ret = the short's % return after all costs (borrow at --borrow-pct);
ret_hard also charges a hard-to-borrow fee of 50% a year. sum = total of ret in
position sizes (1.0 = one whole position made); max_dd = worst drop in that
running total, trades taken in exit order at equal size.

HOW TO RUN:
  python spike_short.py --start 2010-01-01                   # the grid + by year for the pick
  python spike_short.py --start 2010-01-01 --pick 10:100:gap:25 --sample 10
  python spike_short.py --start 2010-01-01 --hold 5
  python spike_short.py --start 2010-01-01 --mins 30,50 --stops 0,20
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import datastore
import engine
import engine_short
import rules
from params import add_param_flags, add_run_flags, collect
from reports import tstat_overlap

RUN_BARS = [3, 10]
RUN_MINS = [50, 100]
TRIGGERS = ["close", "gap"]
STOPS = [0, 15, 25, 40]          # % above the short price, 0 = no stop
FIRST_IN = 10                    # a setup counts only if it didn't also fire in the last 10 bars
HARD_BORROW_PCT = 50.0


def shift_bool(arr, k, sym):
    """Boolean arr shifted k rows earlier, False where it would cross a symbol."""
    out = np.zeros(len(arr), dtype=bool)
    out[k:] = arr[:-k] & (sym[k:] == sym[:-k])
    return out


def run_gain(panel, k):
    """% the close is above the lowest close of the k bars before it (NaN early on)."""
    c, sym = panel.c, panel.sym
    low = np.full(len(c), np.nan)
    for j in range(1, k + 1):
        prev = np.full(len(c), np.nan)
        prev[j:] = np.where(sym[j:] == sym[:-j], c[:-j], np.nan)
        low = prev if j == 1 else np.fmin(low, prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c / low - 1) * 100


def spike_candidates(panel, keep, gain, min_gain, trigger):
    """Entries for one setup in engine.find_entries' format.
    gain: run_gain(panel, K); keep: the universe on each row."""
    sym, o, h, c = panel.sym, panel.o, panel.h, panel.c
    in_run = gain >= min_gain
    if trigger == "close":
        cand = in_run
    else:
        prev_ok = np.zeros(len(c), dtype=bool)
        prev_ok[1:] = sym[1:] == sym[:-1]
        cand = np.zeros(len(c), dtype=bool)
        cand[1:] = in_run[:-1] & (o[1:] > h[:-1])
        cand &= prev_ok
    seen = np.zeros(len(c), dtype=bool)
    for k in range(1, FIRST_IN + 1):
        seen |= shift_bool(cand, k, sym)
    t = np.flatnonzero(cand & ~seen)
    if trigger == "close":
        t = t[keep[t]]
        sig, entry_row, raw, at_close = t, t, c[t], np.ones(len(t), dtype=bool)
    else:
        t = t[keep[t - 1]]                     # judged at yesterday's close, before the open
        sig, entry_row, raw, at_close = t - 1, t, o[t], np.zeros(len(t), dtype=bool)
    ok = np.isfinite(raw) & (raw > 0)
    return {"signal_row": sig[ok], "entry_row": entry_row[ok], "entry_raw": raw[ok],
            "struct_stop": np.full(int(ok.sum()), np.nan), "at_close": at_close[ok]}


def exit_params(stop, hold, borrow_pct):
    return {**engine_short.EXIT_DEFAULTS, "exit": "time", "time_bars": hold,
            "max_hold": max(hold, 60), "stop_mode": "pct",
            "stop_pct": stop if stop > 0 else 1e6,           # "no stop" = one that can't be hit
            "max_risk_pct": 1e9, "borrow_pct": borrow_pct}


def summarize(t, years, hold, borrow_pct):
    if t.empty:
        return {"trades": 0}
    days = (t["exit_date"] - t["entry_date"]).dt.days
    r = t["ret_pct"]
    hard = r - (HARD_BORROW_PCT - borrow_pct) * days / 365.0
    run = t.sort_values("exit_date")["ret_pct"].cumsum() / 100
    by_year = r.groupby(t["entry_date"].dt.year).sum()
    daily = t.groupby("entry_date")["ret_pct"].mean()
    tt, _ = tstat_overlap(daily, hold) if len(daily) > 3 * hold else (None, 0)
    return {
        "trades": len(t), "per_yr": len(t) / years,
        "win_%": (r > 0).mean() * 100, "ret": r.mean(), "median": r.median(),
        "worst_5%": r.quantile(0.05), "worst": r.min(),
        "stopped_%": t["exit_reason"].str.startswith("stopped").mean() * 100,
        "ret_hard": hard.mean(), "sum": r.sum() / 100, "max_dd": (run - run.cummax()).min(),
        "yrs_up": f"{int((by_year > 0).sum())}/{len(by_year)}", "t": tt,
    }


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    add_run_flags(parser)
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--hold", type=int, default=10, help="Bars to hold if not stopped")
    parser.add_argument("--borrow-pct", type=float, default=engine_short.EXIT_DEFAULTS["borrow_pct"],
                        help="Borrow fee, %% a year (ret_hard adds a 50%%/yr stress on top)")
    parser.add_argument("--mins", type=str, default=",".join(str(x) for x in RUN_MINS),
                        help="Run sizes to test, %%, comma separated")
    parser.add_argument("--stops", type=str, default=",".join(str(x) for x in STOPS),
                        help="Stops to test, %% above the short price, 0 = none")
    parser.add_argument("--pick", type=str, default="3:50:close:25",
                        help="K:X:trigger:stop for the by-year table, e.g. 10:100:gap:25")
    parser.add_argument("--save-trades", type=str, default=None, help="CSV path for the pick's trades")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check")
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()
    P = collect(args, ["universe"])
    t0 = time.time()

    cols = (["open", "high", "low", "close", "unadj_close"] + rules.universe_columns(P)
            + engine_short.exit_columns({"exit": "time"}))
    try:
        df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=120)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    keep = rules.universe_mask(panel.df, P)
    if rules.market_columns(P):
        keep &= rules.market_mask(panel.df["date"], datastore.load_market(rules.market_columns(P)), P)
    if args.end is not None:
        keep &= (panel.df["date"] <= pd.Timestamp(args.end)).to_numpy()
    d = panel.df.loc[keep, "date"]
    years = max(0.1, (d.max() - d.min()).days / 365.25) if len(d) else 0.1

    pk = args.pick.split(":")
    pick = (int(pk[0]), float(pk[1]), pk[2], float(pk[3]))
    gains = {k: run_gain(panel, k) for k in RUN_BARS + ([pick[0]] if pick[0] not in RUN_BARS else [])}
    rows, pick_trades = [], None
    mins = [float(x) for x in args.mins.split(",")]
    stops = [float(x) for x in args.stops.split(",")]
    combos = [(k, x, trg, s) for k in RUN_BARS for x in mins for trg in TRIGGERS for s in stops]
    if pick not in combos:
        combos.append(pick)
    for k, x, trg, s in combos:
        cand = spike_candidates(panel, keep, gains[k], x, trg)
        t = engine_short.run_signals(panel, cand, exit_params(s, args.hold, args.borrow_pct),
                                     args.start, args.end, family=f"{k}:{x:g}:{trg}:{s:g}")
        rows.append({"bars": k, "gain>=": x, "entry": trg, "stop": s or "none",
                     **summarize(t, years, args.hold, args.borrow_pct)})
        if (k, x, trg, s) == pick:
            pick_trades = t

    print("=" * 78)
    print(f"SPIKE SHORTS  hold={args.hold}  universe={args.universe}  start={args.start}  end={args.end}  "
          f"liq_top_pct={P['liq_top_pct']:g}  ({time.time() - t0:.1f}s)")
    print("=" * 78)
    print("bars/gain>= : the close is gain% above the lowest close of the `bars` bars before it. "
          "entry: close = short that close; gap = short the next open if it is above that day's high.")
    print("ret = % a trade after slippage, commission, borrow and dividends; ret_hard adds 50%/yr borrow; "
          "worst_5% = the 1-in-20 loss; sum / max_dd in position sizes.")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if pick_trades is None or pick_trades.empty:
        print(f"\nNo trades for the pick {args.pick}")
        return
    t = pick_trades
    y = t.groupby(t["entry_date"].dt.year)["ret_pct"]
    tab = pd.DataFrame({"trades": y.size(), "win_%": y.apply(lambda r: (r > 0).mean() * 100),
                        "ret": y.mean(), "worst": y.min(), "sum": y.sum() / 100})
    print(f"\nPick {args.pick} (bars:gain:entry:stop) by year:")
    print(tab.round(2).to_string())
    print("\nExit reasons:")
    g = t.groupby("exit_reason")["ret_pct"]
    print(pd.DataFrame({"trades": g.size(), "ret": g.mean()}).round(2).to_string())

    if args.save_trades:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_trades)), exist_ok=True)
        t.to_csv(args.save_trades, index=False)
    if args.sample:
        s = t.sample(min(args.sample, len(t)), random_state=args.sample_seed).sort_values("entry_date")
        p = exit_params(pick[3], args.hold, args.borrow_pct)
        slip, comm = p["slippage_pct"] / 100.0, p["commission"]
        view = pd.DataFrame({
            "symbol": s["symbol"], "shorted": s["entry_date"].dt.date,
            "at": np.where(s["at_close"], "close", "open"),
            "short_px": ((s["entry"] + comm) / (1 - slip)).round(2),
            "stop": s["stop"].round(2) if pick[3] > 0 else np.nan,
            "covered": s["exit_date"].dt.date, "cover_px": ((s["exit"] - comm) / (1 + slip)).round(2),
            "reason": s["exit_reason"], "ret_%": s["ret_pct"].round(1)})
        print(f"\n{len(s)} random trades to check on a chart (split-adjusted prices; "
              f"python show_data.py SYMBOL --around DATE):")
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
