"""
Crash-keeps-falling short — real trades with stops
==================================================
crash_short.py found that stocks which ran up 20%+ over 60 days and then fell
30%+ in 3 days kept falling: the short made about +3.2% a trade after costs
over the next 10 days (t 2.8, about 50 a year on Dev since 2010). Those tables
had no stop, and about 1 trade in 3 went 25%+ against the short at some point
in the 10 days. This script trades the same rule through the short engine
(engine_short.simulate_short, as spike_short.py does) with stops, so the
squeezes are capped and every cost is charged: slippage, commission, borrow and
dividends.

The rule (frozen in crash_short.py; the entry, the stop and the filters below are open):
  event   the first day a stock's 3-day close-to-close fall reaches 30%, with no
          such day in the 10 bars before, passing the universe that day ($5
          printed price, top 80% by dollar volume, no recent price break)
  filter  the stock rose 20%+ over the 60 bars before the fall
  entry   close     short that close
          next_open short the next open
          bounce    short a relief rally into a resistance level within the
                    --bounce-bars bars after the crash (the first Dev run found next
                    opens 10%+ above the crash close made +17% a trade on 44 trades).
                    --bounce-level sets the level, known at the crash-day close:
                      10    10% above the crash-day close
                      r38   38% of the 3-day fall won back (between the crash close
                            and the close the fall started from)
                      prev  the close the day before the crash day
                    --bounce-fill sets how it is shorted:
                      open  only when a day opens at or above the level (at that open)
                      touch a short limit order resting at the level: filled at the
                            open if it opens above, else at the level if the day's
                            high reaches it
                    The bounce table compares the levels and both fills at the pick's stop.
  exit    a stop, else cover at the close 10 bars after the entry. Stops:
          25        intraday, 25% above the short price (filled at the open if the
                    stock gaps through it)
          25c       judged on the close: cover at the next open after a close 25%+
                    above the short price (intraday squeezes that fade don't count)
          0         none
Grid: the three entries x stops none / 25 / 40 / 60 / 100 / 25c / 40c / 60c.

Filters with a reason behind them, as flags (all off by default), and stacked one
by one in the filter table for the pick:
  --skip-spike 20   no live spike-short setup on the stock in the 20 bars up to the
                    crash (Mark's rule: trade different stocks from the spike short)
  --min-vol-x 2     crash-day volume 2x+ its usual (heavy selling = real news)
  --market above200 the S&P above its 200-day (in a falling market crashed stocks
                    bounce with it: the crash rebound study's 2020 and 2022)
  --need-200d       200 days of history before the fall (young listings squeeze)

ret = the short's % return after all costs (borrow at --borrow-pct); ret_b100
charges 100% a year borrow instead (hard to borrow). sum = total of ret in
position sizes; max_dd = worst drop in that running total; t = t-stat of ret
with trades grouped by entry month.

Spike-short overlap: each trade is marked if the live spike rule (10-bar 100%+
run, an open 2%+ above that day's high) had a setup on the same stock in the 20
bars up to the crash (the same move) or in the year before.

HOW TO RUN:
  python crash_short/crash_trades.py --start 2010-01-01
  python crash_short/crash_trades.py --start 2010-01-01 --pick next_open:40c --sample 10
  python crash_short/crash_trades.py --start 2010-01-01 --skip-spike 20 --min-vol-x 2 --market above200 --need-200d
  python crash_short/crash_trades.py --start 2010-01-01 --pick bounce:25c --bounce-level r38 --bounce-fill touch
  python crash_short/crash_trades.py --start 2010-01-01 --stop-slip 5
  python crash_short/crash_trades.py --start 2010-01-01 --sizes 2.5,5,10    # account test, alone and with the spike short

The holdout is run once, after the entry, the stop and the filters are chosen on Dev:
  python crash_short/crash_trades.py --start 2010-01-01 --pick ENTRY:STOP [filters] --universe holdout --unlock-holdout
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crash_rebound"))

import datastore  # noqa: E402
import engine  # noqa: E402
import engine_short  # noqa: E402
import research_log  # noqa: E402
import rules  # noqa: E402
import spike_short as ss  # noqa: E402
from params import add_param_flags, add_run_flags, collect  # noqa: E402
import crash_events as ce  # noqa: E402

DAYS, FALL, RUN_MIN = 3, 30.0, 20.0      # the frozen rule
ENTRIES = ["close", "next_open", "bounce"]
STOPS = "0,25,40,60,100,25c,40c,60c"     # % above the short price; c = judged on the close; 0 = none
BOUNCE_LEVELS = ["5", "10", "15", "20", "r25", "r38", "r50", "prev"]
BOUNCE_FILLS = ["open", "touch"]
HARD_BORROW_PCT = 100.0
SPIKE_RULE = {"bars": 10, "gain": 100.0, "min_gap": 2.0, "stop": 25.0, "hold": 10}   # spike_scan.RULE
OVERLAP_BARS = [20, 252]
PASS_BAR = ("Holdout pass bar, set before the holdout is run: the pick's ret above +1.5% a trade at "
            "5%/yr borrow, t above 1.5, and positive in at least 2 of the 2010-14 / 2015-19 / 2020-24 blocks.")


def crash_events(panel, keep, days=DAYS, fall=FALL, run_min=RUN_MIN):
    """The rule's events (rows of panel), with what was known at that close."""
    rows = np.flatnonzero(ce.fall_events(panel.c, panel.sym, days, fall) & keep)
    ev = pd.DataFrame({"row": rows, "fall": ce.fall_pct(panel.c, panel.sym, days)[rows]})
    ev = ce.add_splits(ev, panel.df, panel.sym, rows, days)
    return ev[ev["run60"] >= run_min].reset_index(drop=True)


def bounce_level(panel, sig, level, days=DAYS):
    """The resistance level for each crash row, known at its close: '10' = 10% above
    the crash close, 'r38' = 38% of the fall won back, 'prev' = the prior day's close."""
    c = panel.c
    level = str(level)
    if level == "prev":
        return c[sig - 1]
    if level.startswith("r"):
        return c[sig] + (c[sig - days] - c[sig]) * float(level[1:]) / 100.0
    return c[sig] * (1 + float(level) / 100.0)


def crash_candidates(panel, ev, entry, level="10", fill="open", bars=5):
    """The events as engine entries: short the event close, the next open, or a
    bounce into `level` within `bars` bars (fill: open = only on an open at or above
    it; touch = a limit order at it). signal_row is always the crash day."""
    sig = ev["row"].to_numpy()
    if entry == "close":
        e, raw, at_close = sig, panel.c[sig], np.ones(len(sig), dtype=bool)
    elif entry == "next_open":
        sig = sig[panel.has_next[sig]]                  # no next bar: the stock stopped trading
        e, raw, at_close = sig + 1, panel.o[sig + 1], np.zeros(len(sig), dtype=bool)
    else:
        line = bounce_level(panel, sig, level)
        last = panel.last_row[sig]
        e, raw = np.full(len(sig), -1), np.full(len(sig), np.nan)
        for j in range(1, bars + 1):
            k = np.minimum(sig + j, len(panel.o) - 1)
            ok = (e < 0) & (sig + j <= last)
            at_open = ok & (panel.o[k] >= line)
            at_line = ok & ~at_open & (fill == "touch") & (panel.h[k] >= line)
            e[at_open | at_line] = k[at_open | at_line]
            raw[at_open] = panel.o[k][at_open]
            raw[at_line] = line[at_line]
        keep = e >= 0
        sig, e, raw = sig[keep], e[keep], raw[keep]
        at_close = np.zeros(len(sig), dtype=bool)
    ok = np.isfinite(raw) & (raw > 0)
    return {"signal_row": sig[ok], "entry_row": e[ok], "entry_raw": raw[ok],
            "struct_stop": np.full(int(ok.sum()), np.nan), "at_close": at_close[ok]}


def parse_stop(spec):
    """'25' -> (25.0, False) intraday; '25c' -> (25.0, True) judged on the close; '0' -> no stop."""
    spec = str(spec).strip()
    on_close = spec.endswith("c")
    return float(spec[:-1] if on_close else spec), on_close


def close_stop(panel, t, pct, p):
    """Re-exit no-stop trades with a stop judged on the close: cover at the next
    open after a close pct% or more above the short price (the open entry's own
    close counts). Trades that never close there keep their exit. Costs as in
    engine_short.simulate_short. (Which later setups a stock was too busy to take
    was decided on the no-stop exits; crash setups are 10+ bars apart, so this
    rarely matters.)"""
    if t.empty or pct <= 0:
        return t
    t = t.copy()
    c, o = panel.c, panel.o
    unadj, div = panel.col("unadj_close"), panel.col("div")
    slip, comm = p["slippage_pct"] / 100.0, p["commission"]
    t["stop"] = t["entry"] * (1 + pct / 100.0)
    for i, e, x, at_close, proceeds, raw in zip(t.index, t["entry_row"], t["exit_row"], t["at_close"],
                                                t["entry"], t["entry_raw"]):
        first = e + 1 if at_close else e
        hit = np.flatnonzero(c[first:x] >= raw * (1 + pct / 100.0))
        if not len(hit):
            continue
        k = first + hit[0] + 1                          # cover at the next bar's open
        px = o[k] * (1 + slip) + (comm * c[k] / unadj[k] if unadj[k] > 0 else comm)
        div_paid = float(np.nansum(div[e + 1:k + 1]))
        days = (panel.date[k] - panel.date[e]) / np.timedelta64(1, "D")
        borrow = raw * p["borrow_pct"] / 100.0 * days / 365.0
        pnl = proceeds - px - div_paid - borrow
        t.loc[i, ["exit_row", "exit", "div_paid", "borrow_paid", "ret_pct", "bars_held"]] = \
            [k, px, div_paid, borrow, pnl / proceeds * 100, k - e]
        t.loc[i, "exit_reason"] = "stopped (close)"
        t.loc[i, "exit_date"] = panel.date[k]
    t["exit_row"] = t["exit_row"].astype(int)
    t["result_R"] = (t["entry"] - t["exit"] - t["div_paid"] - t["borrow_paid"]) / (t["stop"] - t["entry"])
    return t


def add_entry_raw(t, cand):
    """The price each short was sold at before costs (the open, the close or the bounce level)."""
    if not t.empty:
        t["entry_raw"] = t["entry_row"].map(dict(zip(cand["entry_row"], cand["entry_raw"])))
    return t


def trade(panel, ev, entry, spec, args, level=None, fill=None):
    """Mode B trades for one entry and stop spec (bounce level and fill from args
    unless given). Adds entry_raw, the price the short was sold at before costs."""
    pct, on_close = parse_stop(spec)
    cand = crash_candidates(panel, ev, entry, level or args.bounce_level, fill or args.bounce_fill,
                            args.bounce_bars)
    p = ss.exit_params(0 if on_close else pct, args.hold, args.borrow_pct, args.slippage_pct)
    t = add_entry_raw(engine_short.run_signals(panel, cand, p, args.start, args.end, family=f"{entry}:{spec}"),
                      cand)
    t = close_stop(panel, t, pct, p) if on_close else ss.apply_stop_slip(t, args.stop_slip)
    return t, cand


def count_before(flag, sym, rows, bars):
    """True values of flag on the same stock in the `bars` bars up to and including each row."""
    cum = np.cumsum(flag.astype(np.int64))
    starts = np.flatnonzero(np.r_[True, sym[1:] != sym[:-1]])
    first = starts[np.searchsorted(starts, rows, side="right") - 1]
    lo = np.maximum(rows - bars + 1, first)
    return cum[rows] - np.where(lo > 0, cum[np.maximum(lo - 1, 0)], 0)


def window_mean(x, sym, ends, n):
    """Mean of x over the n rows ending at each of ends, NaN where that reaches
    before the stock's first row."""
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.asarray(x, np.float64)))])
    starts = np.flatnonzero(np.r_[True, sym[1:] != sym[:-1]])
    safe = np.clip(ends, 0, len(sym) - 1)
    first = starts[np.searchsorted(starts, safe, side="right") - 1]
    lo = ends - n + 1
    ok = (ends >= 0) & (lo >= first)
    return np.where(ok, (cs[safe + 1] - cs[np.clip(lo, 0, len(sym))]) / n, np.nan)


def spike_setups(panel, keep):
    """Entry rows of the live spike-short rule, as a per-row flag."""
    gain = ss.run_gain(panel, SPIKE_RULE["bars"])
    cand = ss.spike_candidates(panel, keep, gain, SPIKE_RULE["gain"], "gap", SPIKE_RULE["min_gap"])
    flag = np.zeros(len(panel.c), dtype=bool)
    flag[cand["entry_row"]] = True
    return flag, cand


def add_event_cols(ev, panel, spike_flag, spx_above200, days=DAYS):
    """What the filters and the extra table rows need, per event."""
    rows = ev["row"].to_numpy()
    ev = ev.copy()
    ev["spike20"] = count_before(spike_flag, panel.sym, rows, 20)
    ev["spx_up"] = spx_above200.reindex(panel.date[rows]).to_numpy() == 1
    st = rows - days                                     # the close the fall is measured from
    vol = panel.col("volume")
    with np.errstate(invalid="ignore", divide="ignore"):
        base = window_mean(vol, panel.sym, st - 20, 100)
        ev["vol_build"] = np.where(base > 0, window_mean(vol, panel.sym, st, 20) / base, np.nan)
        ev["above_lo52"] = (panel.c[rows] / panel.col("lo52_c")[rows] - 1) * 100
    return ev


FILTERS = [   # (name, test), stacked in this order in the filter table; the flags apply the same tests
    ("no spike-short setup in the 20 bars before", lambda e: e["spike20"] == 0),
    ("crash-day volume 2x+ usual", lambda e: e["vol_ratio"] >= 2),
    ("S&P above its 200-day", lambda e: e["spx_up"]),
    ("200 days of history before the fall", lambda e: e["above200_before"].notna()),
]


def summarize(t, years, hold, borrow_pct):
    s = ss.summarize(t, years, hold, borrow_pct)
    s.pop("ret_hard", None)
    if t.empty:
        return s
    days = (t["exit_date"] - t["entry_date"]).dt.days
    s["ret_b100"] = (t["ret_pct"] - max(HARD_BORROW_PCT - borrow_pct, 0.0) * days / 365.0).mean()
    return s


GAP_BINS = [-np.inf, -20, -10, -5, 0, np.inf]
GAP_LABELS = ["-20 or worse", "-20 to -10", "-10 to -5", "-5 to 0", "up"]
NEXT_BINS = [-np.inf, -10, -3, 0, 3, 10, 20, 30, np.inf]
NEXT_LABELS = ["-10 or worse", "-10 to -3", "-3 to 0", "0 to 3", "3 to 10", "10 to 20", "20 to 30", "30+"]
BAR_BINS = [0, 1, 2, 3, 5, np.inf]
BAR_LABELS = ["1", "2", "3", "4-5", "6+"]
LO52_BINS = [-np.inf, 0.5, 10, 30, 100, np.inf]
LO52_LABELS = ["at the low", "0-10", "10-30", "30-100", "100+"]


def thirds(x):
    """ss.thirds with infinities treated as missing (one inf breaks the cut points)."""
    x = np.asarray(x, np.float64)
    return ss.thirds(np.where(np.isfinite(x), x, np.nan)).to_numpy()


def entry_features(panel, t, ev, spike_flag, spx_above200):
    """What was known at entry for each trade, bucketed for the squeeze table."""
    e = ev.set_index("row").loc[t["signal_row"].to_numpy()]
    s = t["signal_row"].to_numpy()
    f = pd.DataFrame(index=t.index)
    f["size of the fall %"] = thirds(-e["fall"])
    f["60-day run-up before %"] = thirds(e["run60"])
    f["event-day volume vs usual (x)"] = thirds(e["vol_ratio"])
    if "vol_build" in e:
        f["volume build-up before the fall (x)"] = thirds(e["vol_build"])
        f["crash close above 52-week low %"] = pd.cut(e["above_lo52"].to_numpy(), LO52_BINS, labels=LO52_LABELS)
    f["event-day open vs prior close %"] = pd.cut(e["gap_pct"].to_numpy(), GAP_BINS, labels=GAP_LABELS)
    if not t["at_close"].all():
        nxt = (t["entry_raw"].to_numpy() / panel.c[s] - 1) * 100
        f["short price vs crash close %"] = pd.cut(np.where(t["at_close"], np.nan, nxt), NEXT_BINS,
                                                   labels=NEXT_LABELS)
        f["bars after the crash"] = pd.cut(t["entry_row"].to_numpy() - s, BAR_BINS, labels=BAR_LABELS)
    f["share price $"] = pd.cut(panel.col("unadj_close")[s], ss.PRICE_BINS, right=False,
                                labels=["5-10", "10-20", "20-50", "50+"])
    f["stock vs its 200d before the fall"] = np.select(
        [e["above200_before"].to_numpy() == 1, e["above200_before"].to_numpy() == 0], ["above", "below"], "no 200d")
    f["S&P vs its 200d"] = np.where(spx_above200.reindex(panel.date[s]).to_numpy() == 1, "above", "below")
    f["in an S&P index"] = np.where(e["in_index"].to_numpy(), "yes", "no")
    f["liquidity rank"] = thirds(panel.col("dv_pct")[s])
    f["reverse split in past year"] = np.where(ss.reverse_splits_before(panel, s) > 0, "yes", "no")
    for b in OVERLAP_BARS:
        n = "20 bars" if b == 20 else "year"
        f[f"spike-short setup on it, past {n}"] = np.where(count_before(spike_flag, panel.sym, s, b) > 0,
                                                           "yes", "no")
    return f


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    add_run_flags(parser)
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--hold", type=int, default=10, help="Bars to hold if not stopped")
    parser.add_argument("--borrow-pct", type=float, default=5.0, help="Borrow fee, %% a year")
    parser.add_argument("--stops", type=str, default=STOPS,
                        help="Stops to test, %% above the short price; 25c = judged on the close; 0 = none")
    parser.add_argument("--slippage-pct", type=float, default=None,
                        help="Stress: slippage %% per fill (default = the engine's 0.2)")
    parser.add_argument("--stop-slip", type=float, default=0.0,
                        help="Stress: intraday stops fill this %% above the stop instead of at it")
    parser.add_argument("--pick", type=str, default="next_open:25",
                        help="ENTRY:STOP for the detail tables, e.g. close:25, next_open:40c, bounce:0")
    parser.add_argument("--bounce-level", type=str, default="10",
                        help="Bounce entry level: 10 = 10%% above the crash close, r38 = 38%% of the fall won "
                             "back, prev = the close before the crash day")
    parser.add_argument("--bounce-fill", choices=BOUNCE_FILLS, default="open",
                        help="Bounce entry: open = only on an open at or above the level; touch = a limit "
                             "order resting at the level")
    parser.add_argument("--bounce-bars", type=int, default=5, help="Bounce entry: bars after the crash to wait")
    parser.add_argument("--skip-spike", type=int, default=0,
                        help="Leave out crashes on a stock with a spike-short setup in this many bars before (0 = keep)")
    parser.add_argument("--min-vol-x", type=float, default=0.0,
                        help="Crash-day volume at least this many times its usual (0 = off)")
    parser.add_argument("--need-200d", action="store_true",
                        help="Only stocks with 200 days of history before the fall")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check on a chart")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--save-trades", type=str, default=None, help="CSV path for the pick's trades")
    parser.add_argument("--sizes", type=str, default="",
                        help="Account test: %% of the account per short, comma separated, e.g. 2.5,5,10")
    parser.add_argument("--max-shorts", type=int, default=10, help="Account test: most shorts open at once")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    t0 = time.time()

    cols = (["open", "high", "low", "close", "unadj_close", "volume", "sma200", "lo52_c", "dv_pct",
             "in_sp500", "in_sp400", "in_sp600"] + rules.universe_columns(P)
            + engine_short.exit_columns({"exit": "time"}))
    lead = None if args.start is None else pd.Timestamp(args.start) - pd.Timedelta(days=400)
    try:
        df = datastore.load_panel(cols, start=lead, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=60)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    univ = rules.universe_mask(panel.df, P).copy()
    keep = univ.copy()
    if args.start:
        keep &= (panel.df["date"] >= pd.Timestamp(args.start)).to_numpy()
    if args.end:
        keep &= (panel.df["date"] <= pd.Timestamp(args.end)).to_numpy()
    if rules.market_columns(P):
        keep &= rules.market_mask(panel.df["date"], datastore.load_market(rules.market_columns(P)), P)
    d = panel.df.loc[keep, "date"]
    years = max(0.1, (d.max() - d.min()).days / 365.25) if len(d) else 0.1

    spx = datastore.load_market(["spx_above200"])
    spx = pd.Series(spx["spx_above200"].to_numpy(), index=pd.to_datetime(spx["date"]).to_numpy())
    spike_flag, spike_cand = spike_setups(panel, univ)
    ev_all = add_event_cols(crash_events(panel, keep), panel, spike_flag, spx)
    ev = ev_all
    if args.skip_spike > 0:
        ev = ev[count_before(spike_flag, panel.sym, ev["row"].to_numpy(), args.skip_spike) == 0]
    if args.min_vol_x > 0:
        ev = ev[ev["vol_ratio"] >= args.min_vol_x]
    if args.need_200d:
        ev = ev[ev["above200_before"].notna()]
    ev = ev.reset_index(drop=True)
    on = ([f"no spike-short setup in the {args.skip_spike} bars before"] if args.skip_spike > 0 else []) + \
        ([f"crash-day volume {args.min_vol_x:g}x+ usual"] if args.min_vol_x > 0 else []) + \
        ([f"market {P['market']}"] if P["market"] != "off" else []) + \
        (["200 days of history before the fall"] if args.need_200d else [])

    entry, spec = args.pick.split(":")
    if entry not in ENTRIES:
        raise SystemExit(f"--pick entry must be one of {ENTRIES}, not {entry!r}")
    try:
        bounce_level(panel, np.array([], dtype=int), args.bounce_level)
    except ValueError:
        raise SystemExit(f"--bounce-level must be a % like 10, a share of the fall like r38, or prev, "
                         f"not {args.bounce_level!r}")
    parse_stop(spec)
    specs = [s.strip() for s in args.stops.split(",")]
    combos = [(en, s) for en in ENTRIES for s in specs]
    if (entry, spec) not in combos:
        combos.append((entry, spec))
    rows, t, cand = [], None, None
    for en, s in combos:
        tr, cd = trade(panel, ev, en, s, args)
        pct, on_close = parse_stop(s)
        rows.append({"entry": en, "stop": "none" if pct == 0 else s,
                     **summarize(tr, years, args.hold, args.borrow_pct)})
        if (en, s) == (entry, spec):
            t, cand = tr, cd

    pd.set_option("display.width", 250)
    print("=" * 100)
    print(f"CRASH-KEEPS-FALLING SHORT, REAL TRADES  hold={args.hold}  universe={args.universe}  start={args.start}  "
          f"end={args.end}  liq_top_pct={P['liq_top_pct']:g}  borrow {args.borrow_pct:g}%/yr  "
          f"stop_slip={args.stop_slip:g}%  bounce level {args.bounce_level} fill {args.bounce_fill} within "
          f"{args.bounce_bars} bars  "
          f"({time.time() - t0:.1f}s)")
    print("=" * 100)
    print(f"Rule: first {DAYS}-day fall of {FALL:g}%+ after a 60-day rise of {RUN_MIN:g}%+; short the close, the "
          f"next open or a bounce; cover at the stop or after {args.hold} days. Filters on: "
          f"{'; '.join(on) if on else 'none'}.")
    print("Stops: 25 = intraday 25% above the short; 25c = cover the next open after a close 25%+ above it. "
          "ret = % a trade after slippage, commission, borrow and dividends; ret_b100 at 100%/yr borrow; "
          "worst_5% = the 1-in-20 loss; sum / max_dd in position sizes; t = t-stat with trades grouped by "
          "entry month.")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if t is None or t.empty:
        print(f"\nNo trades for the pick {args.pick}")
        return

    def with_filters(names):
        g = ev_all
        for name, test in FILTERS:
            if name in names:
                g = g[np.asarray(test(g), dtype=bool)]
        tr, _ = trade(panel, g.reset_index(drop=True), entry, spec, args)
        return summarize(tr, years, args.hold, args.borrow_pct)

    names = [n for n, _ in FILTERS]
    rows = [{"filters": "none (the rule as is)", **with_filters([])}]
    rows += [{"filters": f"+ {n}", **with_filters(names[:k + 1])} for k, n in enumerate(names)]
    rows += [{"filters": f"all but: {n}", **with_filters([m for m in names if m != n])} for n in names]
    print(f"\nFilters on the pick {args.pick}, starting from the rule with none of the filter flags (a --market "
          f"filter is already in), stacked one by one "
          f"(each + row keeps the ones above it), then all four with one left out (a filter that matters "
          f"loses the most when it's the one left out).")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if entry == "bounce":
        rows = []
        for lv in BOUNCE_LEVELS + ([args.bounce_level] if args.bounce_level not in BOUNCE_LEVELS else []):
            for fl in BOUNCE_FILLS:
                tr, _ = trade(panel, ev, entry, spec, args, level=lv, fill=fl)
                rows.append({"level": lv, "fill": fl, **summarize(tr, years, args.hold, args.borrow_pct)})
        print(f"\nBounce levels at the pick's stop ({spec}), shorted within {args.bounce_bars} bars of the crash. "
              f"level: 10 = 10% above the crash close; r38 = 38% of the 3-day fall won back; prev = the close "
              f"before the crash day. fill: open = only on an open at or above the level; touch = a limit order "
              f"at the level (filled at the open if it opens above). Filters as the flags say.")
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    t = t.copy()
    t["block"] = ce.block_of(t["entry_date"])
    rows = [{"block": b, **summarize(g, ce.span_years(g["entry_date"]), args.hold, args.borrow_pct)}
            for b, g in t.groupby("block")]
    print(f"\nPick {args.pick} (entry:stop) by 5-year block:")
    print(pd.DataFrame(rows).drop(columns=["per_yr"]).round(2).to_string(index=False))

    y = t.groupby(t["entry_date"].dt.year)["ret_pct"]
    tab = pd.DataFrame({"trades": y.size(), "win_%": y.apply(lambda r: (r > 0).mean() * 100),
                        "ret": y.mean(), "worst": y.min(), "sum": y.sum() / 100})
    print(f"\nPick {args.pick} by year:")
    print(tab.round(2).to_string())
    print("\nExit reasons:")
    g = t.groupby("exit_reason")["ret_pct"]
    print(pd.DataFrame({"trades": g.size(), "ret": g.mean()}).round(2).to_string())
    print(ss.skip_report(panel, cand, t, args.start, args.end))

    feats = entry_features(panel, t, ev, spike_flag, spx)
    near = (feats["spike-short setup on it, past 20 bars"] == "yes").mean() * 100
    year = (feats["spike-short setup on it, past year"] == "yes").mean() * 100
    print(f"\nSpike-short overlap: {near:.1f}% of the pick's trades are on a stock the live spike rule had a setup "
          f"on in the 20 bars up to the crash (the same move), {year:.1f}% in the year before.")
    print(f"\nWhat separates the winners from the squeezes? Pick {args.pick}, all {len(t)} trades "
          f"(ret {t['ret_pct'].mean():.2f}%), split by what was known at entry. Thirds are by count.")
    print(ss.squeeze_table(t, feats).round(2).to_string(index=False))

    if args.sample:
        s = t.sample(min(args.sample, len(t)), random_state=args.sample_seed).sort_values("entry_date")
        p = ss.exit_params(parse_stop(spec)[0], args.hold, args.borrow_pct, args.slippage_pct)
        slip, comm = p["slippage_pct"] / 100.0, p["commission"]
        e = ev.set_index("row").loc[s["signal_row"].to_numpy()]
        short_px = (s["entry"] + comm) / (1 - slip)
        view = pd.DataFrame({
            "symbol": s["symbol"], "crash_day": s["signal_date"].dt.date,
            "fall_%": (-e["fall"]).round(1).to_numpy(), "run60_%": e["run60"].round(1).to_numpy(),
            "shorted": s["entry_date"].dt.date, "at": np.where(s["at_close"], "close", "open"),
            "short_px": short_px.round(2),
            "real_px": (short_px * panel.col("unadj_close")[s["entry_row"]] / panel.c[s["entry_row"]]).round(2),
            "stop": s["stop"].round(2) if parse_stop(spec)[0] > 0 else np.nan,
            "covered": s["exit_date"].dt.date, "cover_px": ((s["exit"] - comm) / (1 + slip)).round(2),
            "reason": s["exit_reason"], "ret_%": s["ret_pct"].round(1)})
        print(f"\n{len(s)} random trades to check on a chart (prices split-adjusted to today, like TradingView; "
              f"real_px = the price it actually traded at that day; python show_data.py SYMBOL --around DATE):")
        print(view.to_string(index=False))

    if args.save_trades:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_trades)), exist_ok=True)
        t.to_csv(args.save_trades, index=False)
        print(f"\nSaved {len(t)} trades to {args.save_trades}")

    if args.sizes:
        spike_t = engine_short.run_signals(
            panel, spike_cand, ss.exit_params(SPIKE_RULE["stop"], SPIKE_RULE["hold"], args.borrow_pct,
                                              args.slippage_pct), args.start, args.end, family="spike")
        books = {"crash short": t, "spike short": spike_t,
                 "both": pd.concat([t, spike_t], ignore_index=True) if len(spike_t) else t}
        start = pd.Timestamp(args.start) if args.start else d.min()
        dates = pd.DatetimeIndex(np.unique(panel.date[panel.date >= max(start, d.min())]))
        rows, years_tab = [], {}
        for name, tr in books.items():
            if tr.empty:
                continue
            for size in [float(x) for x in args.sizes.split(",")]:
                curve, skipped = ss.account_curve(panel, tr, dates, size, args.max_shorts)
                row, by_year = ss.account_summary(curve, tr, size, skipped)
                rows.append({"book": name, "trades": len(tr), **row})
                years_tab[f"{name} {size:g}%"] = by_year
        print(f"\nAccount test, {dates[0].date()} to {dates[-1].date()}: the crash short ({args.pick}) alone, the "
              f"live spike short alone (its frozen rule) and both in one account. Each short is size_% of the "
              f"account when it opens, at most {args.max_shorts} open at once, marked at every close. cagr / max_dd "
              f"/ worst_year in % of the account; skipped = trades with no room; mc_dd = the worst drop with the "
              f"same trades in a random order (median / 1-in-20).")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        print("\nAccount % return by year:")
        print(pd.DataFrame(years_tab).round(1).to_string())

    print(f"\n{PASS_BAR}")

    if not args.no_log:
        research_log.append(
            question=f"Crash-keeps-falling short as real trades, pick {args.pick}, filters: {'; '.join(on) or 'none'}",
            phase="crash_short", split=args.universe, script="crash_short/crash_trades.py",
            params={**vars(args), **P}, combinations=len(combos), runs=len(combos),
            results={"pick": summarize(t, years, args.hold, args.borrow_pct)})


if __name__ == "__main__":
    main()
