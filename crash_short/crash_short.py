"""
Crash-keeps-falling short — do stocks that ran up and then crashed keep falling?
================================================================================
A lead from the crash rebound study (crash_rebound/): buying crashes only paid
in 2020 and 2022, but some crashes kept falling. After a 30%+ fall over 3 days,
stocks that had risen 20%+ over the 60 days before fell about 4% more over the
next 10 days (t -2.8, about 50 a year on Dev). That was found by scanning about
20 splits, so this script tests it properly before anything is trusted.

The rule, stated before the run (the pick):
  event   first day a stock's 3-day close-to-close fall reaches 30%, with no such
          day in the 10 bars before; passes the universe that day ($5 printed price,
          top 80% by dollar volume, no recent price break, not pinned)
  filter  the stock rose 20%+ over the 60 bars before the fall (close 63 bars back
          to close 3 bars back)
  trade   short that close (MOC), cover at the close 10 bars later

Short view of price breaks: a window across a downward break is left out (an
untrusted gain for a short); an upward one counts as a real loss. A stock that
stops trading uses its last listed close.

Columns (stock moves in %, so a short earns minus raw):
  rawH       average move over H days        medH / fellH_% = median / share that fell
  p95        the 10-day move 1 in 20 trades beats (the squeezes)
  max_up     the worst 10-day move against the short
  hit25_%    share whose daily HIGH went 25%+ above the entry within 10 days
             (a 25% stop would have been hit; with a stop the result would differ)
  net        -raw10 minus the round trip and --borrow-pct a year for 10 days
  net_b100   the same at 100% a year borrow (hard to borrow)
  t          month-clustered t-stat of the short's 10-day return

HOW TO RUN:
  python crash_short/crash_short.py --start 2010-01-01
  python crash_short/crash_short.py --start 2010-01-01 --entry next_open
  python crash_short/crash_short.py --start 2010-01-01 --market above200   # S&P above its 200-day only
  python crash_short/crash_short.py --start 2010-01-01 --sample 10
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crash_rebound"))

import config  # noqa: E402
import datastore  # noqa: E402
import research_log  # noqa: E402
import rules  # noqa: E402
from params import add_param_flags, collect  # noqa: E402
import crash_events as ce  # noqa: E402

FALLS = [20, 30, 50]
RUN_MINS = [None, 0, 20, 50]     # None = no filter on the 60-day move before the fall
HORIZONS = ce.HORIZONS
HARD_BORROW = 100.0


def max_up(df, codes, rows, entry, hold):
    """Highest daily high within `hold` bars after entry, as a fraction above the entry price.
    Close entry: bars row+1..row+hold. Open entry: the entry bar row+1 counts too."""
    h = df["high"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    last = pd.Series(np.arange(len(df))).groupby(codes).transform("max").to_numpy()[rows]
    px = c[rows] if entry == "close" else np.where(rows + 1 <= last, o[np.minimum(rows + 1, len(o) - 1)], np.nan)
    best = np.full(len(rows), np.nan)
    for k in range(1, hold + 1):
        j = rows + k
        ok = j <= last
        hk = np.where(ok, h[np.minimum(j, len(h) - 1)], np.nan)
        best = np.fmax(best, hk)
    return best / px - 1


def short_stats(g, years, hold, borrow_pct):
    out = {"events": len(g), "per_yr": len(g) / years}
    if len(g) == 0:
        return out
    for hz in [5, hold, 20]:
        out[f"raw{hz}"] = g[f"r{hz}"].mean() * 100
    r = g[f"r{hold}"].dropna()
    cost = ce.round_trip_cost()
    out[f"med{hold}"] = r.median() * 100
    out[f"fell{hold}_%"] = (r < 0).mean() * 100
    out["p95"] = r.quantile(0.95) * 100
    out["max_up"] = r.max() * 100
    out["hit25_%"] = (g["maxup"] >= 0.25).mean() * 100
    out["net"] = -out[f"raw{hold}"] - cost - borrow_pct * hold / 252
    out["net_b100"] = -out[f"raw{hold}"] - cost - HARD_BORROW * hold / 252
    out["t"] = ce.tstat_by_month(-g[f"r{hold}"], g["date"])
    out["delist_%"] = g["delisted"].mean() * 100
    return out


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--start", type=str, default=None, help="First event date")
    parser.add_argument("--end", type=str, default=None, help="Last event date")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Only once the rule is frozen.")
    parser.add_argument("--entry", choices=["close", "next_open"], default="close")
    parser.add_argument("--days", type=int, default=3, help="Fall measured over this many days")
    parser.add_argument("--fall", type=float, default=30, help="Pick: fall of at least this %%")
    parser.add_argument("--run-min", type=float, default=20,
                        help="Pick: 60-day move before the fall of at least this %%")
    parser.add_argument("--hold", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument("--borrow-pct", type=float, default=5.0, help="Borrow fee, %% a year")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check on a chart")
    parser.add_argument("--save-events", type=str, default=None, help="Write the pick's events to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    hold, days = args.hold, args.days

    t0 = time.time()
    cols = (rules.universe_columns(P) + ["open", "high", "low", "close", "volume", "sma200",
                                         "bad_bar", "bad_bar_dn", "in_sp500", "in_sp400", "in_sp600"])
    try:
        lead = None if args.start is None else pd.Timestamp(args.start) - pd.Timedelta(days=150)
        df = datastore.load_panel(cols, start=lead, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=40)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python pipeline/build_data.py, then python pipeline/build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy()
    c = df["close"].to_numpy(np.float64)
    market = pd.read_parquet(config.MARKET_FILE, columns=["date", "spx_open", "spx_close"])

    keep = rules.universe_mask(df, P).copy()
    if args.start:
        keep &= (df["date"] >= pd.Timestamp(args.start)).to_numpy()
    if args.end:
        keep &= (df["date"] <= pd.Timestamp(args.end)).to_numpy()
    if rules.market_columns(P):
        keep &= rules.market_mask(df["date"], datastore.load_market(rules.market_columns(P)), P)
    years = ce.span_years(df.loc[keep, "date"]) if keep.any() else 0.1

    spx200 = datastore.load_market(["spx_above200"])

    def events_for(fall):
        rows = np.flatnonzero(ce.fall_events(c, codes, days, fall) & keep)
        ev = pd.DataFrame({"row": rows, "symbol": df["symbol"].to_numpy()[rows],
                           "date": df["date"].to_numpy()[rows],
                           "price": df["unadj_close"].to_numpy(np.float64)[rows],
                           "fall": ce.fall_pct(c, codes, days)[rows]})
        for k, v in ce.forward(df, codes, rows, market, args.entry, side="short").items():
            ev[k] = v
        ev["maxup"] = max_up(df, codes, rows, args.entry, hold)
        ev = ev.merge(spx200, on="date", how="left")
        ev = ce.add_splits(ev, df, codes, rows, days)
        ev["drop"] = -ev["fall"]
        return ev

    evs = {f: events_for(f) for f in sorted(set(FALLS + [args.fall]))}

    print("=" * 110)
    print(f"CRASH-KEEPS-FALLING SHORT  entry={args.entry}  hold={hold}  universe={args.universe}  "
          f"start={args.start}  end={args.end}  market={P['market']}  liq top {P['liq_top_pct']:g}%  "
          f"min price ${P['min_price']:g}  borrow {args.borrow_pct:g}%/yr  ({time.time() - t0:.1f}s)")
    print("=" * 110)
    pd.set_option("display.width", 250)
    cost = ce.round_trip_cost()

    rows = []
    for f in FALLS:
        for rm in RUN_MINS:
            ev = evs[f]
            g = ev if rm is None else ev[ev["run60"] >= rm]
            rows.append({"fall>=": f, "run60>=": "any" if rm is None else rm,
                         **short_stats(g, years, hold, args.borrow_pct)})
    print(f"\n{days}-day falls of X%+, by the 60-day move before the fall. Stock moves in % (a short earns "
          f"minus raw); net = -raw{hold} minus a {cost:.2f}% round trip and {args.borrow_pct:g}%/yr borrow; "
          f"net_b100 at 100%/yr.")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    ev = evs[args.fall]
    pick = ev[ev["run60"] >= args.run_min].reset_index(drop=True)
    print(f"\nPICK: {days}-day fall of {args.fall:g}%+ after a 60-day move of {args.run_min:g}%+, short the "
          f"{'close' if args.entry == 'close' else 'next open'}, cover after {hold} days.")
    if pick.empty:
        print("No trades.")
        return

    splits = [s for s in ce.SPLITS if not s[0].startswith("dropped")] + \
        ce.thirds(pick, "vol_ratio", "event-day volume vs usual") + \
        ce.thirds(pick, "run60", "60-day move before the fall") + ce.thirds(pick, "drop", "size of the fall %")
    show = ["split", "events", "per_yr", "raw5", f"raw{hold}", "raw20", f"med{hold}", f"fell{hold}_%",
            "p95", "hit25_%", "net", "net_b100", "t"]
    rows = [{"split": n, **short_stats(pick.loc[np.asarray(fn(pick), dtype=bool)], years, hold,
                                       args.borrow_pct)} for n, fn in splits]
    print(f"\nPick split by what was known at the entry:")
    print(pd.DataFrame(rows).reindex(columns=show).round(2).to_string(index=False))

    pick["block"] = ce.block_of(pick["date"])
    rows = [{"block": b, **short_stats(g, ce.span_years(g["date"]), hold, args.borrow_pct)}
            for b, g in pick.groupby("block")]
    print(f"\nPick by 5-year block:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    rows = []
    for y, g in pick.groupby(pick["date"].dt.year):
        s = short_stats(g, 1.0, hold, args.borrow_pct)
        rows.append({"year": y, **{k: s.get(k) for k in ["events", f"raw{hold}", f"med{hold}",
                                                         f"fell{hold}_%", "hit25_%", "net"]},
                     "sum_net": s["events"] * s["net"] / 100})
    print(f"\nPick by year (sum_net = total of the shorts after cost, in position sizes):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.sample:
        s = pick.sample(min(args.sample, len(pick)), random_state=1).sort_values("date")
        s = s.assign(fall=s["fall"].round(1), run60=s["run60"].round(1), price=s["price"].round(2),
                     move10=(s[f"r{hold}"] * 100).round(1), max_up=(s["maxup"] * 100).round(1))
        print(f"\n{len(s)} random pick trades to check on a chart (price = printed close on the event day; "
              f"move10 = what the stock did, the short earns minus it):")
        print(s[["symbol", "date", "fall", "run60", "price", "move10", "max_up"]].to_string(index=False))

    if args.save_events:
        pick.drop(columns=["row"]).to_csv(args.save_events, index=False)
        print(f"\nSaved {len(pick)} trades to {args.save_events}")

    print(f"\nWorth building on = net clearly above 0 with t above 2, net_b100 not far below 0, and positive "
          f"in most years and in the 2010-2014 / 2015-2019 / 2020-2024 blocks.")

    if not args.no_log:
        s = short_stats(pick, years, hold, args.borrow_pct)
        research_log.append(
            question="Do stocks that ran up 20%+ and then fell 30%+ in 3 days keep falling?",
            phase="crash_short", split=args.universe, script="crash_short/crash_short.py",
            params={**P, "start": args.start, "end": args.end, "entry": args.entry, "days": days,
                    "fall": args.fall, "run_min": args.run_min, "hold": hold, "borrow_pct": args.borrow_pct},
            combinations=len(FALLS) * len(RUN_MINS), runs=1,
            results={k: (round(float(v), 3) if v is not None and np.isfinite(v) else None)
                     for k, v in s.items()})


if __name__ == "__main__":
    main()
