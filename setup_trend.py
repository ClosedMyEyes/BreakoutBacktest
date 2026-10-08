"""
Modes B and C — signal test and portfolio test
==============================================
Mode B (--mode signal): every entry signal taken, one position per symbol at a
a time, results in R. Always run next to F0, the date-matched random
baseline: random stocks that passed the same filters on the same days, with
the same risk and the same exits. The edge is avg_R minus F0's avg_R.

Mode C (--mode portfolio): the Mode B trades run through an account with
capital, max positions, risk per trade and same-day ranking.

HOW TO RUN:
  python setup_trend.py                                   # F1 50-day breakout, X2 exit, Dev
  python setup_trend.py --family f2 --k 10 --t 8 --exit x3 --trail ema21
  python setup_trend.py --mode portfolio --max-positions 8 --risk-pct 0.75
  python setup_trend.py --template off --n 20 --exit time --time-bars 20 --stop-pct 0
                                                          # Phase 4 dummy strategy
  python setup_trend.py --save-trades results/trades.csv  # trade log for hand checks

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
import rules
from params import add_param_flags, add_run_flags, collect

parser = argparse.ArgumentParser()
add_param_flags(parser, ["universe", "template", "entry", "exit", "portfolio"])
add_run_flags(parser)
parser.add_argument("--mode", choices=["signal", "portfolio"], default="signal")
parser.add_argument("--f0-seeds", type=int, default=3, help="Random baseline repeats (0 = skip F0)")
parser.add_argument("--save-trades", type=str, default=None, help="CSV path for the trade log")
parser.add_argument("--monte-carlo", type=int, default=0, help="Trade-order reshuffles for a drawdown range")
args = parser.parse_args()

P = collect(args, ["universe", "template", "entry", "exit", "portfolio"])
TAIL_DAYS = 500   # load this many days past --end so late trades can exit


def main():
    t0 = time.time()
    cols = (["open", "high", "low", "close"] + rules.universe_columns(P) + rules.template_columns(P)
            + rules.entry_columns(P) + engine.exit_columns(P) + ["rs_rank", "tight_10"])
    df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                              unlock_holdout=args.unlock_holdout, tail_days=TAIL_DAYS)
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    df = panel.df
    if not args.output_json:
        print(f"Loaded {len(df):,} rows, {df['symbol'].nunique():,} symbols ({time.time() - t0:.1f}s)")

    base = rules.universe_mask(df, P) & rules.template_mask(df, P)
    setups = rules.entry_setups(df, P, base)
    cand = engine.find_entries(panel, setups, P)
    trades = engine.run_signals(panel, cand, P, args.start, args.end, family=P["family"])
    stats = engine.signal_metrics(trades)

    # F0: date-matched random baseline, several seeds
    f0_avgs = []
    for seed in range(args.f0_seeds):
        f0 = engine.run_f0(panel, base, trades, P, seed=seed)
        if not f0.empty:
            f0_avgs.append(f0["result_R"].mean())
    if f0_avgs:
        stats["f0_avg_R"] = round(float(np.mean(f0_avgs)), 3)
        stats["f0_min"] = round(float(np.min(f0_avgs)), 3)
        stats["f0_max"] = round(float(np.max(f0_avgs)), 3)
        stats["edge_R"] = round(stats["avg_R"] - stats["f0_avg_R"], 3)

    if args.monte_carlo and len(trades):
        stats.update(engine.monte_carlo_dd(trades.sort_values("entry_date")["result_R"],
                                           P["risk_pct"], n=args.monte_carlo))

    curve = None
    if args.mode == "portfolio":
        market = datastore.load_market(["spx_close"])
        curve, taken = engine.run_portfolio(panel, trades, P, market)
        pstats = engine.portfolio_metrics(curve, market)
        stats = {**pstats, "signals": stats["trades"], "taken": int(len(taken)),
                 "signal_avg_R": stats["avg_R"], "edge_R": stats.get("edge_R")}
        if args.save_trades:
            trades = taken

    if args.save_trades and len(trades):
        os.makedirs(os.path.dirname(os.path.abspath(args.save_trades)), exist_ok=True)
        trades.to_csv(args.save_trades, index=False)

    if args.output_json:
        print("GRID_RESULT:" + json.dumps(stats, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
        return

    print("=" * 60)
    print(f"MODE {'B: SIGNAL' if args.mode == 'signal' else 'C: PORTFOLIO'} TEST  "
          f"family={P['family']}  exit={P['exit']}  universe={args.universe}")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<16}: {v}")
    if args.mode == "signal" and len(trades):
        print("\nExit reasons:")
        print(trades["exit_reason"].value_counts().to_string())
        print("\nBy year:")
        print(engine.yearly_table(trades).to_string())
    if curve is not None and len(curve):
        eq = curve.set_index("date")["equity"]
        year_end = eq.groupby(eq.index.year).last()
        start = year_end.shift(1).fillna(P["capital"])
        yearly = ((year_end / start - 1) * 100).round(1)
        yearly.index.name = "year"
        print("\nPortfolio return by year (%, first and last years are partial):")
        print(yearly.to_string())
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
