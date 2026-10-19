"""
Market-wide gap-downs, part 2 — where the trades lose, and filters on the stock itself
=====================================================================================
breadth_events.py's pick failed on Dev (Mark's run, 2026-10-19): 2.5-5% gap-downs of S&P 500 members on broad
days, bought at the open and sold at the close, lost 0.11% a trade after costs from 2010 on (t -0.48) and were
positive in 6 of 15 years. The losses came from crash days, when 20%+ of members gapped down: 50 days since
2010 with 4,844 trades at -0.37%, while the 5-20% days made +0.35%. Before 2010 the crash days did best, and the
locked half (other companies on the same days) can't check a rule about which days to trade. It can check a
rule about the stock, so this shows where the losses are and splits the events by a few things about the stock.

Same events, breadth and costs as breadth_events.py (the same flags), on any breadth, broad and lone days side
by side.

  BY YEAR        broad and lone days before 2010 (breadth_events.py prints 2010 on)
  WORST AND      the days whose events add up to the largest loss or gain, all years, with the day's breadth
  BEST DAYS      and SPY's gap and open-to-close; then 2010 on without the 10 worst and without the 10 best
  FILTERS        each known at the open, and why it might matter:
    trend        the prior close against the stock's 200-day and 50-day averages. A gap-down in a rising stock
                 may get bought back; in a falling one it may keep going.
    below SPY    how many points further the stock gapped down than SPY that morning. A gap close to the
                 market's is mostly the market's move; one far below it is mostly the stock's own news, which
                 tends to keep going.
    vs its range the gap over the stock's average true range (50 days, as % of the prior close). 3% is a shock
                 for a steady stock and an ordinary day for a jumpy one.
    day before   the stock's close-to-close move the day before. A second fall in a row may be panic that
                 turns, or a slide that carries on.

The bar, set before the first run: a bucket is worth one run on the locked half if, on all days from 2010, its
ret is above 0 with a month-clustered t above 2, and above 0 in each 5-year block from 2010. With 13 buckets one
can clear t 2 by luck, and the locked half shares the days, so a pass there would still need the forward test.

Columns: trades, ret (% a trade after --cost-pct) and t (month-clustered) on all days from 2010; broad_ and
lone_ the same on broad and lone days; the 5-year blocks and "before 2010" are ret on all days.

HOW TO RUN:
  python gapdown_breadth/filters.py
  python gapdown_breadth/filters.py --save-trades gapdown_filters.csv    # every event with its filter values
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth_events as be  # noqa: E402  (puts core/ and crash_rebound/ on the path)
import crash_events as ce  # noqa: E402
import research_log  # noqa: E402
from params import collect  # noqa: E402

N_DAYS = 15         # worst and best days listed
N_DROP = 10         # 2010 on without this many worst / best days
FEATURES = ["sma50", "sma200", "atr50"]


def spy_moves(market):
    """SPY's gap at the open (against the prior close) and its open-to-close, in %, by date."""
    if not {"spy_open", "spy_close"} <= set(market.columns):
        return pd.DataFrame({"spy_gap": [], "spy_oc": []}, index=pd.DatetimeIndex([], name="date"))
    m = market.sort_values("date")
    o = m["spy_open"].where(m["spy_open"] > 0).to_numpy(np.float64)
    c = m["spy_close"].where(m["spy_close"] > 0).to_numpy(np.float64)
    prev = np.r_[np.nan, c[:-1]]
    return pd.DataFrame({"spy_gap": (o / prev - 1) * 100, "spy_oc": (c / o - 1) * 100},
                        index=pd.DatetimeIndex(m["date"], name="date"))


def stock_features(df, codes, t, moves):
    """What each event's stock looked like at the open: the prior close against its 200- and 50-day averages
    (%), how many points further it gapped down than SPY, the gap over its 50-day ATR (as % of the prior
    close), and its close-to-close move the day before (%)."""
    p = t["row"].to_numpy() - 1
    c = df["close"].to_numpy(np.float64)
    gap = t["gap"].to_numpy(np.float64)
    spy_gap = moves["spy_gap"].reindex(pd.DatetimeIndex(t["date"])).to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        vs200 = (c[p] / df["sma200"].to_numpy(np.float64)[p] - 1) * 100
        vs50 = (c[p] / df["sma50"].to_numpy(np.float64)[p] - 1) * 100
        atr_pct = df["atr50"].to_numpy(np.float64)[p] / c[p] * 100
        gap_atr = np.where(atr_pct > 0, -gap / atr_pct, np.nan)
    return t.assign(vs200=vs200, vs50=vs50, spy_gap=spy_gap, below_spy=spy_gap - gap, gap_atr=gap_atr,
                    day_before=ce.fall_pct(c, codes, 1)[p])


def bins(v, edges, labels):
    """(label, mask) for each [lo, hi) of v; NaN is in none."""
    v = np.asarray(v, dtype=np.float64)
    return [(lab, (v >= a) & (v < b)) for lab, a, b in zip(labels, edges[:-1], edges[1:])]


def filter_groups(t):
    """The filters' buckets, (group, [(bucket, mask)]), after a row for every event."""
    vs = lambda col: [("above", t[col].to_numpy() > 0), ("below", t[col].to_numpy() <= 0)]
    return [
        ("every event", [("", np.ones(len(t), dtype=bool))]),
        ("trend: vs its 200-day", vs("vs200")),
        ("trend: vs its 50-day", vs("vs50")),
        ("gap below SPY's, points", bins(t["below_spy"], [-np.inf, 1, 2.5, np.inf],
                                         ["under 1", "1 to 2.5", "2.5+"])),
        ("gap over its 50-day ATR", bins(t["gap_atr"], [0, 1, 2, np.inf], ["under 1x", "1x to 2x", "2x+"])),
        ("the day before, %", bins(t["day_before"], [-np.inf, -2, 2, np.inf], ["fell 2+", "-2 to +2", "rose 2+"])),
    ]


def num(x):
    return np.nan if x is None else float(x)


def filter_table(t, groups, broad, split_year=be.SPLIT_YEAR):
    """For each bucket: trades, ret and t on all days from split_year, the same on broad and lone days, ret on
    all days in each 5-year block from split_year and before it, and whether it clears the bar (ret above 0
    with t above 2, and above 0 in every block)."""
    late = pd.DatetimeIndex(t["date"]).year.to_numpy() >= split_year
    wide = (t["breadth"] >= broad).to_numpy()
    blk = be.block(t["date"])
    blocks = sorted(set(blk[late]))
    rows = []
    for group, buckets in groups:
        for label, m in buckets:
            m = np.asarray(m, dtype=bool)
            r = {"group": group, "bucket": label}
            for pre, mm in [("", m & late), ("broad_", m & late & wide), ("lone_", m & late & ~wide)]:
                st = be.stats(t[mm])
                r.update({f"{pre}trades": st["trades"], f"{pre}ret": st.get("ret", np.nan),
                          f"{pre}t": num(st.get("t"))})
            for b in blocks:
                g = t["ret"][m & late & (blk == b)]
                r[b] = g.mean() if len(g) else np.nan
            g = t["ret"][m & ~late]
            r[f"before {split_year}"] = g.mean() if len(g) else np.nan
            r["passes"] = bool(r["ret"] > 0 and r["t"] > 2 and all(r[b] > 0 for b in blocks))
            rows.append(r)
    return pd.DataFrame(rows)


def day_totals(t, moves):
    """One row per day with events: its breadth, how many events, their average and summed ret (the summed ret
    is the day's result in % of one position, as if each trade were the same size), and SPY's gap and
    open-to-close."""
    d = t.groupby("date").agg(breadth=("breadth", "first"), events=("ret", "size"), avg_ret=("ret", "mean"),
                              total=("ret", "sum"))
    return d.join(moves, how="left")


def without_extremes(days, n=N_DROP, split_year=be.SPLIT_YEAR):
    """From split_year: the events' total and ret a trade, then without the n worst and the n best days."""
    d = days[pd.DatetimeIndex(days.index).year >= split_year]
    out = []
    for name, keep in [("every day", d), (f"without the {n} worst days", d.drop(d.nsmallest(n, "total").index)),
                       (f"without the {n} best days", d.drop(d.nlargest(n, "total").index))]:
        out.append({"days": name, "events": int(keep["events"].sum()), "total": keep["total"].sum(),
                    "ret": keep["total"].sum() / keep["events"].sum() if keep["events"].sum() else np.nan})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    be.add_event_flags(parser)
    parser.add_argument("--days", type=int, default=N_DAYS, help="Worst and best days to list (15)")
    parser.add_argument("--save-trades", type=str, default=None,
                        help="Write every event, with its filter values, to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])

    t0 = time.time()
    df, codes, market, gap, brk, br, spy, t = be.load_events(args, P, FEATURES)
    moves = spy_moves(market)
    t = stock_features(df, codes, t, moves)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("=" * 110)
    print(f"MARKET-WIDE GAP-DOWNS: WHERE THEY LOSE, AND FILTERS ON THE STOCK  events: {args.min_gap:g}-"
          f"{args.max_gap:g}% gap-downs, size={P['size']}  broad = {args.broad:g}%+  universe={args.universe}  "
          f"start={args.start}  end={args.end}  cost {args.cost_pct:g}%  ({time.time() - t0:.1f}s)")
    print("=" * 110)
    if t.empty:
        print("No events.")
        return
    split = be.SPLIT_YEAR
    y = pd.DatetimeIndex(t["date"]).year
    print(f"{len(t):,} events on {t['date'].nunique():,} days, {t['date'].min():%Y-%m-%d} to "
          f"{t['date'].max():%Y-%m-%d}; {int((y >= split).sum()):,} of them from {split}.")

    pre = t[y < split]
    spy_pre = spy[pd.DatetimeIndex(spy["date"]).year < split] if len(spy) else spy
    print(f"\nBY YEAR, BEFORE {split} (broad = events on broad days, lone = on the other days, spy_broad_ret = SPY "
          f"on broad days; breadth_events.py has {split} on):")
    print(be.by_period(pre[pre["breadth"] >= args.broad], pre[pre["breadth"] < args.broad],
                       spy_pre[spy_pre["breadth"] >= args.broad] if len(spy_pre) else spy_pre, be.year)
          .round(2).to_string(index=False))

    days = day_totals(t, moves)
    cols = ["breadth", "events", "avg_ret", "total", "spy_gap", "spy_oc"]
    print(f"\nTHE {args.days} WORST DAYS, all years (total = the day's events' ret added up, in % of one position; "
          "spy_gap and spy_oc = SPY's gap at the open and its open-to-close, %):")
    print(days.nsmallest(args.days, "total")[cols].round(2).to_string())
    print(f"\nTHE {args.days} BEST DAYS, all years:")
    print(days.nlargest(args.days, "total")[cols].round(2).to_string())
    print(f"\n{split} ON, WITHOUT THE WORST OR THE BEST DAYS (ret = % a trade):")
    print(without_extremes(days, N_DROP, split).round({"total": 1, "ret": 3}).to_string(index=False))

    table = filter_table(t, filter_groups(t), args.broad, split)
    n_buckets = len(table) - 1
    print(f"\nFILTERS ON THE STOCK, {split} on, all days and split into broad and lone days; the 5-year blocks and "
          f"before {split} are ret on all days. The first row is every event.")
    print(table.round(2).to_string(index=False))
    passing = table.iloc[1:][table["passes"].iloc[1:]]
    print(f"\nThe bar, set before the first run: ret above 0 with t above 2 on all days from {split}, and above 0 in "
          f"every 5-year block from {split}. "
          + (f"Clearing it: {'; '.join(passing['group'] + ' ' + passing['bucket'])}, out of {n_buckets} buckets; "
             "each is a candidate for one run on the locked half, not a result."
             if len(passing) else f"None of the {n_buckets} buckets clears it."))

    if args.save_trades:
        t.drop(columns=["row"]).to_csv(args.save_trades, index=False)
        print(f"\nSaved {len(t):,} events with their filter values to {args.save_trades}")

    if not args.no_log:
        rnd = lambda v: round(float(v), 4) if v is not None and np.isfinite(v) else None
        research_log.append(
            question="Do filters on the stock (trend, gap below SPY's, gap over ATR, the day before) turn the "
                     "gap-down open-to-close trade positive?",
            phase="gapdown_breadth", split=args.universe, script="filters.py",
            params={"start": args.start, "end": args.end, "min_gap": args.min_gap, "max_gap": args.max_gap,
                    "broad": args.broad, "min_members": args.min_members, "cost_pct": args.cost_pct,
                    "size": P["size"], "liq_top_pct": P["liq_top_pct"], "min_price": P["min_price"]},
            combinations=n_buckets, runs=1,
            results={"buckets": [{"group": r["group"], "bucket": r["bucket"], "trades": int(r["trades"]),
                                  "ret": rnd(r["ret"]), "t": rnd(r["t"]), "passes": bool(r["passes"])}
                                 for _, r in table.iterrows()]})


if __name__ == "__main__":
    main()
