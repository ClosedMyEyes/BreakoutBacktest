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
A target (--target) covers early with a buy limit order below the short, as a
multiple of the distance from the short price down to the crash low (the lowest
low of the 3 crash days, known at the crash close):
  0.5     half way down to the crash low
  1       at the crash low
  1.5     half that distance again below the crash low
  0       none
It is checked from the day after the entry: filled at the open if it opens at or
below the line, else at the line if the day's low reaches it. On the day a stop
covers, the stop is assumed to come first.

Tables, for --pick LEVEL:FILL:WINDOW:STOP, --target and --hold:
  every level x fill x window (--windows) at the pick's stop, target and hold
  the pick's entry with each stop (--stops)
  the pick's entry with each target (--targets) x each hold (--holds)
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
  python crash_bounce/bounce_short.py --start 2010-01-01 --target 0.9 --hold 20 --sizes 5
  python crash_bounce/bounce_short.py --start 2010-01-01 --skip-spike 20

The holdout is run once, after the pick, the hold and any filters are chosen on Dev:
  python crash_bounce/bounce_short.py --start 2010-01-01 --pick LEVEL:FILL:WINDOW:STOP --target X --hold N [filters] --universe holdout --unlock-holdout
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
TOUCH_MARGINS = [0.5, 1.0, 2.0]   # the fill check: % the high must clear the line by
TARGETS = "0,0.5,0.9,1,1.5"   # multiples of the distance from the short price to the crash low; 0 = none
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


def crash_low(panel, sig, days=ct.DAYS):
    """The lowest low of the crash days (the wick), known at the crash close."""
    return np.min([panel.l[sig - k] for k in range(days)], axis=0)


def take_profit(panel, t, lines, p):
    """Re-exit trades with a buy limit order at each trade's line, from the bar after
    the entry: filled at the open if it opens at or below the line, else at the line
    if the day's low reaches it. A time exit's last bar is checked; on a stop's last
    bar the stop comes first."""
    if t.empty:
        return t
    t = t.copy()
    idx, ks, prices = [], [], []
    for i, e, x, reason, line in zip(t.index, t["entry_row"], t["exit_row"], t["exit_reason"], lines):
        last = x if reason == "time exit" else x - 1
        hit = np.flatnonzero(panel.l[e + 1:last + 1] <= line)
        if len(hit):
            k = e + 1 + hit[0]
            idx.append(i)
            ks.append(k)
            prices.append(min(panel.o[k], line))
    return ct.cover(panel, t, idx, np.asarray(ks, dtype=int), prices, p, "target")


def bounce_trades(panel, ev, level, fill, bars, stop, hold, args, target=0.0, margin=0.0):
    """The bounce short's trades for one level, fill, window, stop, hold and target.
    margin: a touch fill at the line counts only when the high clears it by this %."""
    cand = ct.crash_candidates(panel, ev, "bounce", level, fill, bars, margin)
    family = f"{level}:{fill}:{bars}:{stop}"
    p = ss.exit_params(0, hold, args.borrow_pct, args.slippage_pct)
    if stop != PRE:
        t = ct.run_trades(panel, cand, stop, hold, args, family)
    else:
        t = ct.add_entry_raw(engine_short.run_signals(panel, cand, p, args.start, args.end, family=family), cand)
        if not t.empty:
            lines = panel.c[t["signal_row"].to_numpy() - ct.DAYS]   # the close the fall started from
            t = ct.close_stop_at(panel, t, lines, p, "stopped (crash undone)")
    if target > 0 and not t.empty:
        raw = t["entry_raw"].to_numpy()
        t = take_profit(panel, t, raw - target * (raw - crash_low(panel, t["signal_row"].to_numpy())), p)
    return t, cand


def target_name(x):
    return "none" if x == 0 else f"{x:g}x"


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
    parser.add_argument("--target", type=float, default=0.0,
                        help="Cover target for the pick, a multiple of the distance from the short price to the "
                             "crash low: 0.5 = half way, 1 = the crash low, 1.5 = beyond it; 0 = none")
    parser.add_argument("--targets", type=str, default=TARGETS, help="Targets to test on the pick")
    parser.add_argument("--touch-margin", type=float, default=0.0,
                        help="Stress: a touch fill at the line counts only when the day's high clears the line by "
                             "this %% (still filled at the line)")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    t0 = time.time()
    try:
        level, fill, bars, stop = parse_pick(args.pick)
        stops = [check_stop(s.strip()) for s in args.stops.split(",")]
        windows = sorted({int(w) for w in args.windows.split(",")} | {bars})
        holds = sorted({int(h) for h in args.holds.split(",")} | {args.hold})
        targets = sorted({float(x) for x in args.targets.split(",")} | {args.target})
    except ValueError:
        raise SystemExit(f"--pick is LEVEL:FILL:WINDOW:STOP, e.g. r38:touch:5:0: level a % like 10, a share of "
                         f"the fall like r38, or prev; fill {' or '.join(FILLS)}; window 1 or more bars; stop "
                         f"0, 25, 25c or pre (--stops, --windows and --holds take the same values, comma "
                         f"separated; --targets like 0,0.5,1). Got --pick {args.pick}.")
    stops += [stop] if stop not in stops else []

    D = ct.load_crashes(args, P)
    panel, ev, years = D.panel, D.ev, D.years

    def run(lv=level, fl=fill, br=bars, st=stop, hold=args.hold, target=args.target, events=None,
            margin=args.touch_margin):
        return bounce_trades(panel, ev if events is None else events, lv, fl, br, st, hold, args, target, margin)

    def exits(t):
        if t.empty:
            return {}
        return {"target_hit_%": (t["exit_reason"] == "target").mean() * 100, "avg_bars": t["bars_held"].mean()}

    def summ(t, hold=args.hold):
        return ct.summarize(t, years, hold, args.borrow_pct)

    levels = LEVELS + ([level] if level not in LEVELS else [])
    grid = pd.DataFrame([{"level": lv, "fill": fl, "window": br, **summ(run(lv, fl, br)[0])}
                         for lv in levels for fl in FILLS for br in windows])
    by_stop = pd.DataFrame([{"stop": "none" if s == "0" else s, **summ(run(st=s)[0])} for s in stops])
    by_exit = []
    for h in holds:
        for x in targets:
            tr = run(hold=h, target=x)[0]
            by_exit.append({"hold": h, "target": target_name(x), **summ(tr, h), **exits(tr)})
    by_exit = pd.DataFrame(by_exit)
    pick = args.pick + (f" target {target_name(args.target)}" if args.target else "") + \
        (f" touch margin {args.touch_margin:g}%" if args.touch_margin else "")
    t, cand = run()
    fills = None
    if fill == "touch" and not t.empty:
        at_open = np.isclose(t["entry_raw"].to_numpy(), panel.o[t["entry_row"].to_numpy()])
        fills = [{"fills": "all, as run", **summ(t)},
                 {"fills": "at the open (it opened at or above the line)", **summ(t[at_open])},
                 {"fills": "at the line (the day's high reached it)", **summ(t[~at_open])}]
        fills += [{"fills": f"count a fill at the line only if the high cleared it by {m:g}%+",
                   **summ(run(margin=m)[0])} for m in TOUCH_MARGINS if m > args.touch_margin]
        fills = pd.DataFrame(fills)

    pd.set_option("display.width", 250)
    print("=" * 100)
    print(f"CRASH BOUNCE SHORT  pick={args.pick}  target={target_name(args.target)}  hold={args.hold}  "
          f"universe={args.universe}  start={args.start}  "
          f"end={args.end}  liq_top_pct={P['liq_top_pct']:g}  borrow {args.borrow_pct:g}%/yr  "
          f"stop_slip={args.stop_slip:g}%  ({time.time() - t0:.1f}s)")
    print("=" * 100)
    print(f"Rule: first {ct.DAYS}-day fall of {ct.FALL:g}%+ after a 60-day rise of {ct.RUN_MIN:g}%+; short the "
          f"relief rally into a line known at the crash close, within the window; cover at the stop, the "
          f"target or after the hold. Filters on: {'; '.join(D.on) if D.on else 'none'}.")
    print("level: 10 / 20 = that % above the crash close; r38 = 38% of the 3-day fall won back; prev = the close "
          "before the crash day. fill: open = only on an open at or above the line; touch = a limit order at the "
          "line (filled at the open if it opens above). window = bars after the crash the line has to be reached "
          "in. ret = % a trade after slippage, commission, borrow and dividends; ret_b100 at 100%/yr borrow; "
          "worst_5% = the 1-in-20 loss; sum / max_dd in position sizes; t = t-stat with trades grouped by entry "
          "month.")
    print(f"\nEvery level, fill and window, stop {'none' if stop == '0' else stop}, target "
          f"{target_name(args.target)}, hold {args.hold}:")
    print(grid.round(2).to_string(index=False))
    print(f"\nStops on {level}:{fill}:{bars}, target {target_name(args.target)}, hold {args.hold}. 25 = intraday "
          f"25% above the short; 25c = cover the next open after a close 25%+ above it; pre = cover the next open "
          f"after a close back at or above the close the 3-day fall started from.")
    print(by_stop.round(2).to_string(index=False))
    print(f"\nTargets x holds on {level}:{fill}:{bars}, stop {'none' if stop == '0' else stop}. target = a buy "
          f"limit below the short at that multiple of the distance from the short price to the crash low (the "
          f"lowest low of the 3 crash days): 0.5x half way, 1x the crash low, 1.5x beyond it. hold = bars after "
          f"the entry to cover at the close if the target isn't hit (target none = the plain time exit). "
          f"target_hit_% = share covered at the target; avg_bars = bars held.")
    print(by_exit.round(2).to_string(index=False))
    if fills is not None:
        print(f"\nTouch fill check on the pick: is the edge in fills that only just reached the line? The margin rows "
              f"count a fill at the line only when the day's high cleared it by that much (still filled at the "
              f"line), as if a limit order resting at the line had not been reached in the queue.")
        print(fills.round(2).to_string(index=False))

    if t.empty:
        print(f"\nNo trades for the pick {pick}")
        return

    print(f"\nFilters on the pick {pick}, starting from the rule with none of the filter flags (a --market "
          f"filter is already in), stacked one by one (each + row keeps the ones above it), then all four with "
          f"one left out (a filter that matters loses the most when it's the one left out).")
    print(ct.filter_rows(D, lambda g: run(events=g)[0], years, args).round(2).to_string(index=False))

    ct.detail_tables(D, t, cand, args, pick, has_stop(stop))
    if args.sizes:
        ct.account_test(D, t, args, "bounce short", f"the bounce short ({pick}, hold {args.hold})")

    print(f"\n{PASS_BAR}")

    if not args.no_log:
        runs = len(grid) + len(by_stop) + len(by_exit) + 1 + (len(fills) - 3 if fills is not None else 0)
        research_log.append(
            question=f"Crash bounce short, pick {pick} hold {args.hold}, filters: {'; '.join(D.on) or 'none'}",
            phase="crash_bounce", split=args.universe, script="crash_bounce/bounce_short.py",
            params={**vars(args), **P}, combinations=runs, runs=runs,
            results={"pick": summ(t)})


if __name__ == "__main__":
    main()
