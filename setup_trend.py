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
  python setup_trend.py --sample 10                       # 10 random trades to check on charts
  python setup_trend.py --sample 10 --sample-reason delisted
  python setup_trend.py --family f3 --sample 5 --sample-result win    # 5 random winners
  python setup_trend.py --family f3 --market trend        # only while the S&P 500 is in an uptrend

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
from reports import block_table, exit_table

parser = argparse.ArgumentParser()
add_param_flags(parser, ["universe", "template", "entry", "exit", "portfolio"])
add_run_flags(parser)
parser.add_argument("--mode", choices=["signal", "portfolio"], default="signal")
parser.add_argument("--f0-seeds", type=int, default=3, help="Random baseline repeats (0 = skip F0)")
parser.add_argument("--save-trades", type=str, default=None, help="CSV path for the trade log")
parser.add_argument("--monte-carlo", type=int, default=0, help="Trade-order reshuffles for a drawdown range")
parser.add_argument("--sample", type=int, default=0, help="Print N random trades to check by hand on a chart")
parser.add_argument("--sample-reason", type=str, default=None,
                    help="Only sample trades whose exit reason contains this text (e.g. delisted)")
parser.add_argument("--sample-result", choices=["all", "win", "loss"], default="all",
                    help="Only sample winning (R > 0) or losing trades")
parser.add_argument("--sample-seed", type=int, default=0, help="Change to draw a different sample")
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
    if rules.market_columns(P):          # market light: no new entries (F0 too) on red days
        base &= rules.market_mask(df["date"], datastore.load_market(rules.market_columns(P)), P)
    setups = rules.entry_setups(df, P, base)
    cand = engine.find_entries(panel, setups, P)
    trades = engine.run_signals(panel, cand, P, args.start, args.end, family=P["family"])
    stats = engine.signal_metrics(trades)

    # F0: date-matched random baseline, several seeds
    f0_avgs, f0_wins, f0_runs = [], [], []
    for seed in range(args.f0_seeds):
        f0 = engine.run_f0(panel, base, trades, P, seed=seed)
        if not f0.empty:
            f0_avgs.append(f0["result_R"].mean())
            f0_wins.append((f0["result_R"] > 0).mean() * 100)
            f0_runs.append(f0)
    f0_first = f0_runs[0] if f0_runs else None
    if f0_avgs:
        stats["f0_avg_R"] = round(float(np.mean(f0_avgs)), 3)
        stats["f0_min"] = round(float(np.min(f0_avgs)), 3)
        stats["f0_max"] = round(float(np.max(f0_avgs)), 3)
        stats["f0_win_rate"] = round(float(np.mean(f0_wins)), 1)
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
        print(f"\nExit reasons ({P['family']}, then F0 seed 0 for comparison):")
        print(exit_table(trades, f0_first).to_string())
        if f0_runs:
            print("\nEdge over F0 by 5-year block (is it there in every period?):")
            print(block_table(trades, pd.concat(f0_runs)).to_string())
        print("\nBy year:")
        print(engine.yearly_table(trades).to_string())
        if args.sample:
            print_sample(trades, df, setups, args.sample, args.sample_reason, args.sample_result,
                         args.sample_seed)
    if curve is not None and len(curve):
        eq = curve.set_index("date")["equity"]
        year_end = eq.groupby(eq.index.year).last()
        start = year_end.shift(1).fillna(P["capital"])
        yearly = ((year_end / start - 1) * 100).round(1)
        yearly.index.name = "year"
        print("\nPortfolio return by year (%, first and last years are partial):")
        print(yearly.to_string())
    print(f"\nDone in {time.time() - t0:.1f}s")


def print_sample(trades, df, setups, n, reason=None, result="all", seed=0):
    """Random trades with the actual fill prices (before slippage and
    commission) so they can be checked on a chart, next to the setup's pivot
    on the signal day (and for F3 its range and volume ratios). Prices are
    split-adjusted."""
    t = trades if reason is None else trades[trades["exit_reason"].str.contains(reason, regex=False)]
    if result != "all":
        t = t[(t["result_R"] > 0) if result == "win" else (t["result_R"] <= 0)]
    if t.empty:
        print(f"\nNo trades match (exit reason {reason!r}, result {result})")
        return
    s = t.sample(min(n, len(t)), random_state=seed).sort_values("entry_date")
    sig = s["signal_row"].to_numpy()
    slip, comm = P["slippage_pct"] / 100.0, P["commission"]
    view = pd.DataFrame({
        "symbol": s["symbol"], "signal_day": s["signal_date"].dt.date,
        "pivot": np.round(setups["pivot"][sig].astype(float), 2),
        "bought": s["entry_date"].dt.date,
        "buy_px": ((s["entry"] - comm) / (1 + slip)).round(2),
        "stop": s["stop"].astype(float).round(2), "sold": s["exit_date"].dt.date,
        "sell_px": ((s["exit"] + comm) / (1 - slip)).round(2),
        "reason": s["exit_reason"], "R": s["result_R"].round(2),
    })
    if P["family"] == "f3":      # how quiet the 10 days before the signal were
        view.insert(3, "range_ratio", df["atr_ratio"].to_numpy()[sig].round(2))
        view.insert(4, "vol_ratio", df["vol10_50"].to_numpy()[sig].round(2))
    print(f"\n{len(s)} random trades to check on a chart (split-adjusted prices; "
          f"delisted stocks: python show_data.py SYMBOL):")
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
