"""
Crash bounce short
==================
Split out of crash_short/crash_trades.py into its own setup. The crash event is
the same; the short is taken on the relief rally after it, at a resistance line
known at the crash-day close. On Dev since 2010 (crash_trades.py, PR #21 run, no
stop, 10-day hold, shorted within 5 bars): r38 open +11.84% a trade (128 trades),
r38 touch +10.02% (235, t 4.56), prev touch +8.54% (300). Shorts on the first day
after the crash made +23% a trade (44 trades), shorts on the third day +0.75%.

The event (frozen in crash_short.py): the first day a stock's 3-day close-to-close
fall reaches 30%, with no such day in the 10 bars before, after a 60-day rise of
20%+, passing the universe that day ($5 printed price, top 80% by dollar volume,
no recent price break).

The entry (chosen here on Dev):
  level   the resistance line, known at the crash-day close
            10, 20        that % above the crash-day close
            r25 r38 r50   that share of the 3-day fall won back
            prev          the close the day before the crash day
  fill    open    short only on a day that opens at or above the line, at the open
          touch   a short limit order resting at the line: filled at the open if
                  it opens above, else at the line if the day's high reaches it
  window  the line has to be reached within this many bars after the crash
          (1 = the next day only)
The exit: a stop, else cover at the close --hold bars after the entry.
  0       none
  25      intraday, 25% above the short price (filled at the open on a gap through)
  25c     cover at the next open after a close 25%+ above the short price
  pre     cover at the next open after a close at or above the close the 3-day
          fall started from (the crash fully undone)

Tables, for --pick LEVEL:FILL:WINDOW:STOP and --hold:
  every level x fill x window (--windows) at the pick's stop and hold
  the pick's entry with each stop (--stops), then with each hold (--holds)
  the filter table (as in crash_trades.py; the filter flags are off by default)
  the pick by block and year, exit reasons, spike-short overlap, and what separates
  the winners from the squeezes
  --sizes: the account test, the bounce short alone, the live spike short alone
  and both in one account

ret = the short's % return after all costs (borrow at --borrow-pct); ret_b100
charges 100% a year borrow instead (hard to borrow). sum = total of ret in
position sizes; max_dd = worst drop in that running total; t = t-stat of ret
with trades grouped by entry month.

HOW TO RUN:
  python crash_bounce/bounce_short.py --start 2010-01-01
  python crash_bounce/bounce_short.py --start 2010-01-01 --pick r38:touch:2:0 --sizes 2.5,5,10
  python crash_bounce/bounce_short.py --start 2010-01-01 --pick prev:touch:5:pre --sample 10
  python crash_bounce/bounce_short.py --start 2010-01-01 --skip-spike 20

The holdout is run once, after the pick, the hold and any filters are chosen on Dev:
  python crash_bounce/bounce_short.py --start 2010-01-01 --pick LEVEL:FILL:WINDOW:STOP --hold N [filters] --universe holdout --unlock-holdout
The pass bar is printed at the end of every run, so it is fixed before that run.
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

import engine_short  # noqa: E402
import research_log  # noqa: E402
import spike_short as ss  # noqa: E402
from params import collect  # noqa: E402
import crash_trades as ct  # noqa: E402

LEVELS = ["10", "20", "r25", "r38", "r50", "prev"]
FILLS = ct.BOUNCE_FILLS
WINDOWS = "1,2,5"
STOPS = "0,25,40,60,100,25c,40c,60c,pre"
HOLDS = "5,10,20"
PICK = "r38:touch:5:0"
PRE = "pre"
PASS_BAR = ("Holdout pass bar for the bounce short, set before the holdout is run: the pick's ret above +5% a "
            "trade at 5%/yr borrow, t above 1.5, and positive in at least 2 of the 2010-14 / 2015-19 / 2020-24 "
            "blocks.")


def check_level(level):
    """ValueError unless level is a % (10), a share of the fall (r38) or prev."""
    if level != "prev":
        float(level[1:] if level.startswith("r") else level)
    return level


def check_stop(stop):
    """ValueError unless stop is pre or a crash_trades stop (0, 25, 25c)."""
    if stop != PRE:
        ct.parse_stop(stop)
    return stop


def parse_pick(spec):
    """'r38:touch:5:0' -> ('r38', 'touch', 5, '0')."""
    parts = str(spec).split(":")
    if len(parts) != 4 or parts[1] not in FILLS:
        raise ValueError(spec)
    level, fill, bars, stop = parts
    bars = int(bars)
    if bars < 1:
        raise ValueError(spec)
    return check_level(level), fill, bars, check_stop(stop)


def has_stop(stop):
    return stop == PRE or ct.parse_stop(stop)[0] > 0


def bounce_trades(panel, ev, level, fill, bars, stop, hold, args):
    """The bounce short's trades for one level, fill, window, stop and hold."""
    cand = ct.crash_candidates(panel, ev, "bounce", level, fill, bars)
    family = f"{level}:{fill}:{bars}:{stop}"
    if stop != PRE:
        return ct.run_trades(panel, cand, stop, hold, args, family), cand
    p = ss.exit_params(0, hold, args.borrow_pct, args.slippage_pct)
    t = ct.add_entry_raw(engine_short.run_signals(panel, cand, p, args.start, args.end, family=family), cand)
    if t.empty:
        return t, cand
    lines = panel.c[t["signal_row"].to_numpy() - ct.DAYS]       # the close the fall started from
    return ct.close_stop_at(panel, t, lines, p, "stopped (crash undone)"), cand


def main():
    parser = argparse.ArgumentParser()
    ct.add_common_flags(parser)
    parser.add_argument("--pick", type=str, default=PICK,
                        help="LEVEL:FILL:WINDOW:STOP for the detail tables, e.g. r38:touch:5:0, prev:open:2:25c, "
                             "r50:touch:1:pre")
    parser.add_argument("--windows", type=str, default=WINDOWS,
                        help="Windows for the level table: bars after the crash the line has to be reached in")
    parser.add_argument("--stops", type=str, default=STOPS,
                        help="Stops to test on the pick: %% above the short price; 25c = judged on the close; "
                             "pre = a close back at the close the fall started from; 0 = none")
    parser.add_argument("--holds", type=str, default=HOLDS, help="Holds to test on the pick, in bars")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    t0 = time.time()
    try:
        level, fill, bars, stop = parse_pick(args.pick)
        stops = [check_stop(s.strip()) for s in args.stops.split(",")]
        windows = sorted({int(w) for w in args.windows.split(",")} | {bars})
        holds = sorted({int(h) for h in args.holds.split(",")} | {args.hold})
    except ValueError:
        raise SystemExit(f"--pick is LEVEL:FILL:WINDOW:STOP, e.g. r38:touch:5:0: level a % like 10, a share of "
                         f"the fall like r38, or prev; fill {' or '.join(FILLS)}; window 1 or more bars; stop "
                         f"0, 25, 25c or pre (--stops, --windows and --holds take the same values, comma "
                         f"separated). Got --pick {args.pick}.")
    stops += [stop] if stop not in stops else []

    D = ct.load_crashes(args, P)
    panel, ev, years = D.panel, D.ev, D.years

    def run(lv=level, fl=fill, br=bars, st=stop, hold=args.hold, events=None):
        return bounce_trades(panel, ev if events is None else events, lv, fl, br, st, hold, args)

    def summ(t, hold=args.hold):
        return ct.summarize(t, years, hold, args.borrow_pct)

    levels = LEVELS + ([level] if level not in LEVELS else [])
    grid = pd.DataFrame([{"level": lv, "fill": fl, "window": br, **summ(run(lv, fl, br)[0])}
                         for lv in levels for fl in FILLS for br in windows])
    by_stop = pd.DataFrame([{"stop": "none" if s == "0" else s, **summ(run(st=s)[0])} for s in stops])
    by_hold = pd.DataFrame([{"hold": h, **summ(run(hold=h)[0], h)} for h in holds])
    t, cand = run()

    pd.set_option("display.width", 250)
    print("=" * 100)
    print(f"CRASH BOUNCE SHORT  pick={args.pick}  hold={args.hold}  universe={args.universe}  start={args.start}  "
          f"end={args.end}  liq_top_pct={P['liq_top_pct']:g}  borrow {args.borrow_pct:g}%/yr  "
          f"stop_slip={args.stop_slip:g}%  ({time.time() - t0:.1f}s)")
    print("=" * 100)
    print(f"Rule: first {ct.DAYS}-day fall of {ct.FALL:g}%+ after a 60-day rise of {ct.RUN_MIN:g}%+; short the "
          f"relief rally into a line known at the crash close, within the window; cover at the stop or after "
          f"the hold. Filters on: {'; '.join(D.on) if D.on else 'none'}.")
    print("level: 10 / 20 = that % above the crash close; r38 = 38% of the 3-day fall won back; prev = the close "
          "before the crash day. fill: open = only on an open at or above the line; touch = a limit order at the "
          "line (filled at the open if it opens above). window = bars after the crash the line has to be reached "
          "in. ret = % a trade after slippage, commission, borrow and dividends; ret_b100 at 100%/yr borrow; "
          "worst_5% = the 1-in-20 loss; sum / max_dd in position sizes; t = t-stat with trades grouped by entry "
          "month.")
    print(f"\nEvery level, fill and window, stop {'none' if stop == '0' else stop}, hold {args.hold}:")
    print(grid.round(2).to_string(index=False))
    print(f"\nStops on {level}:{fill}:{bars}, hold {args.hold}. 25 = intraday 25% above the short; 25c = cover the "
          f"next open after a close 25%+ above it; pre = cover the next open after a close back at or above the "
          f"close the 3-day fall started from.")
    print(by_stop.round(2).to_string(index=False))
    print(f"\nHolds on {level}:{fill}:{bars}, stop {'none' if stop == '0' else stop} (bars after the entry):")
    print(by_hold.round(2).to_string(index=False))

    if t.empty:
        print(f"\nNo trades for the pick {args.pick}")
        return

    print(f"\nFilters on the pick {args.pick}, starting from the rule with none of the filter flags (a --market "
          f"filter is already in), stacked one by one (each + row keeps the ones above it), then all four with "
          f"one left out (a filter that matters loses the most when it's the one left out).")
    print(ct.filter_rows(D, lambda g: run(events=g)[0], years, args).round(2).to_string(index=False))

    ct.detail_tables(D, t, cand, args, args.pick, has_stop(stop))
    if args.sizes:
        ct.account_test(D, t, args, "bounce short", f"the bounce short ({args.pick}, hold {args.hold})")

    print(f"\n{PASS_BAR}")

    if not args.no_log:
        runs = len(grid) + len(by_stop) + len(by_hold) + 1
        research_log.append(
            question=f"Crash bounce short, pick {args.pick} hold {args.hold}, filters: {'; '.join(D.on) or 'none'}",
            phase="crash_bounce", split=args.universe, script="crash_bounce/bounce_short.py",
            params={**vars(args), **P}, combinations=runs, runs=runs,
            results={"pick": summ(t)})


if __name__ == "__main__":
    main()
