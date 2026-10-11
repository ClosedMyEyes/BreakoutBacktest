"""
Short side, Mode B — every short signal vs S0
=============================================
The short-side twin of setup_trend.py, kept separate so no long result moves.
Every signal of one short setup is taken (one position per symbol at a time),
results in R, next to S0: random downtrend-template stocks shorted on the same
days, same time of day, same risk % and same exits. The edge is avg_R minus
S0's avg_R.

Setups (rules_short.py): s1 simple breakdown, s2 bear flag, s3 failed leader,
s4 rally into resistance, s5 quiet breakdown, s6 impulse retest failure, s7 slow
fade. Exit xs2 by default
(engine_short.py). Costs include borrow (--borrow-pct, % a year) and every
dividend paid while short.

HOW TO RUN:
  python setup_short.py                                  # S1 20-day breakdown, S&P below its 200-day
  python setup_short.py --family s1 --n 50
  python setup_short.py --family s5 --market off          # any market (for comparison)
  python setup_short.py --family s3 --s3-v 1.8
  python setup_short.py --family s2 --flag-w 8 --flag-d-pct 30
  python setup_short.py --family s6 --s6-dr-min 0.7          # pullback of at least 70%
  python setup_short.py --family s7 --pool on                # slow fade, downtrend stocks only
  python setup_short.py --borrow-pct 3                    # costlier borrow
  python setup_short.py --save-trades results/s1.csv
  python setup_short.py --sample 10 --sample-result win   # 10 random winners to check on charts

Prints GRID_RESULT:<json> with --output-json (used by grid.py).
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import datastore
import engine
import engine_short
import rules_short
from params import add_param_flags, add_run_flags, collect
from reports import block_table, exit_table

GROUPS = ["s_universe", "s_template", "s_entry", "s_exit"]
parser = argparse.ArgumentParser()
add_param_flags(parser, GROUPS)
add_run_flags(parser)
parser.add_argument("--mode", choices=["signal"], default="signal",
                    help="Only Mode B so far; the short portfolio test comes later")
parser.add_argument("--s0-seeds", "--f0-seeds", dest="s0_seeds", type=int, default=3,
                    help="Random baseline repeats (0 = skip S0)")
parser.add_argument("--save-trades", type=str, default=None, help="CSV path for the trade log")
parser.add_argument("--sample", type=int, default=0, help="Print N random trades to check by hand on a chart")
parser.add_argument("--sample-reason", type=str, default=None,
                    help="Only sample trades whose exit reason contains this text")
parser.add_argument("--sample-result", choices=["all", "win", "loss"], default="all")
parser.add_argument("--sample-seed", type=int, default=0)
args = parser.parse_args()

P = collect(args, GROUPS)
TAIL_DAYS = 200   # load this many days past --end so late shorts can be covered (max_hold 60)

# Columns added for the short side: older data builds don't have them
SHORT_COLUMNS = {"div", "bad_bar_dn", "sma200_dn21", "sma200_dn63", "sma200_dn105", "sma50_dn5",
                 "ema21_dn5", "bflag_bot", "bflag_age", "bflag_high", "bars_since_leader"}


def main():
    t0 = time.time()
    tp = rules_short.base_params(P)
    cols = (["open", "high", "low", "close", "unadj_close"] + rules_short.universe_columns(P)
            + rules_short.template_columns(tp) + rules_short.entry_columns(P)
            + engine_short.exit_columns(P))
    have = set().union(*(datastore._schema_cols(path) for path, is_label in datastore.ROW_FILES
                         if not is_label))
    engine_short.check_columns(have, [c for c in dict.fromkeys(cols)
                                      if c in SHORT_COLUMNS or c.startswith("bpole_high")
                                      or c.startswith("bflag_volr")])
    df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                              unlock_holdout=args.unlock_holdout, tail_days=TAIL_DAYS)
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    df = panel.df
    if not args.output_json:
        print(f"Loaded {len(df):,} rows, {df['symbol'].nunique():,} symbols ({time.time() - t0:.1f}s)")

    base = rules_short.universe_mask(df, P) & rules_short.template_mask(df, tp)
    if rules_short.market_columns(P):      # no new shorts (S0 too) unless the market is weak
        base &= rules_short.market_mask(df["date"], datastore.load_market(rules_short.market_columns(P)), P)
    setups = rules_short.entry_setups(df, P, base)
    cand = engine_short.find_entries(panel, setups, P)
    trades = engine_short.run_signals(panel, cand, P, args.start, args.end, family=P["family"])
    stats = engine.signal_metrics(trades)
    stats.update(engine_short.cost_summary(trades))

    s0_avgs, s0_wins, s0_runs = [], [], []
    for seed in range(args.s0_seeds):
        s0 = engine_short.run_s0(panel, base, trades, P, seed=seed)
        if not s0.empty:
            s0_avgs.append(s0["result_R"].mean())
            s0_wins.append((s0["result_R"] > 0).mean() * 100)
            s0_runs.append(s0)
    if s0_avgs:
        stats["s0_avg_R"] = round(float(np.mean(s0_avgs)), 3)
        stats["s0_min"] = round(float(np.min(s0_avgs)), 3)
        stats["s0_max"] = round(float(np.max(s0_avgs)), 3)
        stats["s0_win_rate"] = round(float(np.mean(s0_wins)), 1)
        stats["edge_R"] = round(stats["avg_R"] - stats["s0_avg_R"], 3)

    if args.save_trades and len(trades):
        os.makedirs(os.path.dirname(os.path.abspath(args.save_trades)), exist_ok=True)
        trades.to_csv(args.save_trades, index=False)

    if args.output_json:
        print("GRID_RESULT:" + json.dumps(stats, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
        return

    print("=" * 60)
    print(f"SHORT MODE B: SIGNAL TEST  setup={P['family']}  exit={P['exit']}  market={P['market']}  "
          f"universe={args.universe}")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<16}: {v}")
    print("  (results are after slippage, commission, borrow and dividends; div_R and borrow_R are "
          "the average cost of each per trade)")
    if len(trades):
        print(f"\nExit reasons ({P['family']}, then S0 seed 0 for comparison):")
        print(exit_table(trades, s0_runs[0] if s0_runs else None, base="s0").to_string())
        if s0_runs:
            print("\nEdge over S0 by 5-year block (is it there in every period?):")
            print(block_table(trades, pd.concat(s0_runs), base="s0").to_string())
        print("\nBy year:")
        print(engine.yearly_table(trades).to_string())
        if args.sample:
            print_sample(trades, setups)
    print(f"\nDone in {time.time() - t0:.1f}s")


def print_sample(trades, setups):
    """Random shorts with the actual fill prices (before slippage and
    commission) to check on a chart. Prices are split-adjusted."""
    t = trades
    if args.sample_reason:
        t = t[t["exit_reason"].str.contains(args.sample_reason, regex=False)]
    if args.sample_result != "all":
        t = t[(t["result_R"] > 0) if args.sample_result == "win" else (t["result_R"] <= 0)]
    if t.empty:
        print("\nNo trades match the sample filters")
        return
    s = t.sample(min(args.sample, len(t)), random_state=args.sample_seed).sort_values("entry_date")
    sig = s["signal_row"].to_numpy()
    slip, comm = P["slippage_pct"] / 100.0, P["commission"]
    view = pd.DataFrame({
        "symbol": s["symbol"], "signal_day": s["signal_date"].dt.date,
        "trigger": np.round(setups["pivot"][sig].astype(float), 2),
        "shorted": s["entry_date"].dt.date,
        "short_px": ((s["entry"] + comm) / (1 - slip)).round(2),
        "stop": s["stop"].astype(float).round(2), "covered": s["exit_date"].dt.date,
        "cover_px": ((s["exit"] - comm) / (1 + slip)).round(2),
        "divs": s["div_paid"].round(2), "reason": s["exit_reason"], "R": s["result_R"].round(2),
    })
    if setups.get("entry") == "signal":          # S3, S4, S6 short the signal day's close
        view = view.drop(columns="trigger")
    print(f"\n{len(s)} random shorts to check on a chart (split-adjusted prices; "
          f"python show_data.py SYMBOL --around DATE):")
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
