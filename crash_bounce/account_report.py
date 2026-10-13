"""
Account report: the crash bounce short and the live spike short in one small account
=====================================================================================
The two frozen rules, each alone and both together, traded through an account that
starts at --capital ($10,000), each short --sizes % of the account when it opens:
  the bounce short   prev:touch:1:0, hold 10, no stop (crash_bounce/bounce_short.py)
  the spike short    its frozen live rule (spike_scan.RULE: 25% stop, hold 10), no judges

Per trade, in % of the position: win rate, average win and loss, payoff (average win
over average loss), the 1-in-20 loss and days held. R = 25% of the position, the spike
short's stop distance, so a stopped spike short is about -1R. The bounce has no stop,
so its losses can run past 1R. $ a trade = the average trade on a position of
--capital x size.

The account: the balance at each year end, the year's return, the worst drop. Open
shorts are marked at every close; costs, borrow and dividends are in every trade.

Time in trades: the share of trading days with a short open, how many are open at
once, and how much of the account is short on average (the rest sits in cash).

HOW TO RUN:
  python crash_bounce/account_report.py --start 2010-01-01
  python crash_bounce/account_report.py --start 2010-01-01 --universe all --unlock-holdout --sizes 5,10
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crash_short"))
sys.path.insert(0, HERE)

import engine  # noqa: E402
import engine_short  # noqa: E402
import research_log  # noqa: E402
import spike_short as ss  # noqa: E402
from params import collect  # noqa: E402
import crash_trades as ct  # noqa: E402
import bounce_short as bs  # noqa: E402

BOUNCE_RULE = bs.FROZEN          # frozen 2026-10-13, holdout passed
R_PCT = 25.0                     # 1R = 25% of the position, the spike short's stop distance


def trade_stats(t, years, capital, sizes):
    """Per-trade numbers for one book, in % of the position and in R, and the
    average trade in $ on a capital x size position for each size."""
    r = t["ret_pct"]
    win, loss = r[r > 0], r[r <= 0]
    avg_loss = loss.mean() if len(loss) else 0.0
    return {"trades": len(t), "per_yr": len(t) / years, "win_%": (r > 0).mean() * 100,
            "avg_win_%": win.mean() if len(win) else 0.0, "avg_loss_%": avg_loss,
            "payoff": win.mean() / -avg_loss if len(win) and avg_loss < 0 else np.nan,
            "ret_%": r.mean(), "worst_5%": r.quantile(0.05), "worst_%": r.min(),
            "avg_R": r.mean() / R_PCT, "avg_win_R": win.mean() / R_PCT if len(win) else 0.0,
            "avg_loss_R": avg_loss / R_PCT, "worst_R": r.min() / R_PCT,
            "days_held": t["bars_held"].mean(),
            **{f"$_a_trade_{s:g}%": capital * s / 100 * r.mean() / 100 for s in sizes}}


def time_in_trades(curve):
    """How much of the time and the account the book has short."""
    held = curve["positions"] > 0
    short = curve["invested"] / curve["equity"] * 100
    return {"days_in_a_trade_%": held.mean() * 100, "avg_open": curve["positions"].mean(),
            "avg_open_when_in": curve.loc[held, "positions"].mean() if held.any() else 0.0,
            "max_open": int(curve["positions"].max()), "avg_short_%": short.mean(),
            "avg_short_%_when_in": short[held].mean() if held.any() else 0.0, "max_short_%": short.max(),
            "avg_cash_%": 100 - short.mean()}


def account_rows(curve, capital):
    """Year-end balance and the year's return."""
    eq = curve.set_index("date")["equity"]
    end = eq.groupby(eq.index.year).last()
    return end, (end / end.shift(1).fillna(capital) - 1) * 100


def drawdown(curve):
    eq = curve["equity"].to_numpy()
    peak = np.maximum.accumulate(eq)
    k = int(np.argmin(eq / peak))
    return (eq[k] / peak[k] - 1) * 100, peak[k] - eq[k], curve["date"].iloc[k]


def main():
    parser = argparse.ArgumentParser()
    ct.add_common_flags(parser)
    parser.set_defaults(sizes="5")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting balance, $")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    t0 = time.time()
    sizes = [float(x) for x in args.sizes.split(",")]

    D = ct.load_crashes(args, P)
    panel = D.panel
    level, fill, bars, stop = bs.parse_pick(BOUNCE_RULE)
    bounce, _ = bs.bounce_trades(panel, D.ev, level, fill, bars, stop, args.hold, args)
    rule = ct.SPIKE_RULE
    spike = engine_short.run_signals(panel, D.spike_cand,
                                     ss.exit_params(rule["stop"], rule["hold"], args.borrow_pct, args.slippage_pct),
                                     args.start, args.end, family="spike")
    books = {"bounce": bounce, "spike": spike,
             "both": pd.concat([bounce, spike], ignore_index=True) if len(spike) else bounce}
    books = {k: v for k, v in books.items() if not v.empty}
    if not books:
        raise SystemExit("No trades in this window.")
    start = pd.Timestamp(args.start) if args.start else D.first_date
    dates = pd.DatetimeIndex(np.unique(panel.date[panel.date >= max(start, D.first_date)]))

    stats = [{"book": name, **trade_stats(t, D.years, args.capital, sizes)} for name, t in books.items()]
    timing, summary, balances, returns = [], [], {}, {}
    for size in sizes:
        for name, t in books.items():
            curve, skipped = ss.account_curve(panel, t, dates, size, args.max_shorts, capital=args.capital)
            m = engine.portfolio_metrics(curve)
            dd_pct, dd_usd, dd_date = drawdown(curve)
            end, ret = account_rows(curve, args.capital)
            label = f"{name} {size:g}%"
            balances[label], returns[label] = end, ret
            summary.append({"book": name, "size_%": size, "start_$": args.capital,
                            "end_$": curve["equity"].iloc[-1], "cagr_%": m["cagr"], "max_dd_%": dd_pct,
                            "max_dd_$": -dd_usd, "dd_low": dd_date.date(), "worst_year_%": ret.min(),
                            "best_year_%": ret.max(), "years_up": f"{int((ret > 0).sum())}/{len(ret)}",
                            "skipped": skipped})
            timing.append({"book": name, "size_%": size, **time_in_trades(curve)})

    pd.set_option("display.width", 250)
    print("=" * 100)
    print(f"ACCOUNT REPORT  bounce {BOUNCE_RULE} hold {args.hold}  spike short (frozen live rule, no judges)  "
          f"universe={args.universe}  {dates[0].date()} to {dates[-1].date()}  start ${args.capital:,.0f}  "
          f"borrow {args.borrow_pct:g}%/yr  at most {args.max_shorts} shorts open  ({time.time() - t0:.1f}s)")
    print("=" * 100)
    print(f"\nPer trade, % of the position. payoff = average win / average loss. R = {R_PCT:g}% of the position "
          f"(the spike short's stop, so a stopped spike short is about -1R; the bounce has no stop). $_a_trade_5% = "
          f"the average trade on a position of 5% of ${args.capital:,.0f}.")
    print(pd.DataFrame(stats).round(2).to_string(index=False))
    print(f"\nThe account from ${args.capital:,.0f}: each short is size_% of the account when it opens, marked at "
          f"every close. max_dd = the worst drop from a high (dd_low = the day it bottomed); skipped = trades "
          f"with {args.max_shorts} shorts already open.")
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    print(f"\nBalance at each year end, $ ({dates[-1].year} runs to {dates[-1].date()}):")
    print(pd.DataFrame(balances).round(0).to_string())
    print(f"\nReturn by year, %:")
    print(pd.DataFrame(returns).round(1).to_string())
    print("\nTime in trades. days_in_a_trade_% = trading days with at least one short open at the close; "
          "avg_open = shorts open on an average day (when_in = on days with one open); avg_short_% = the short "
          "positions' value as % of the account; avg_cash_% = the rest, free for other trades.")
    print(pd.DataFrame(timing).round(2).to_string(index=False))

    if not args.no_log:
        research_log.append(
            question=f"Account report: bounce {BOUNCE_RULE} and the spike short from ${args.capital:,.0f}",
            phase="crash_bounce", split=args.universe, script="crash_bounce/account_report.py",
            params={**vars(args), **P}, combinations=len(summary), runs=len(summary),
            results={"summary": pd.DataFrame(summary).round(2).astype(str).to_dict("records")})


if __name__ == "__main__":
    main()
