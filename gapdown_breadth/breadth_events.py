"""
Market-wide gap-downs — buy the stocks that gap down with the market, at the open, out at the close
===================================================================================================
A lead from the Stockbot review (stockbot_review/verification/README.md, section 9). Stockbot's
gap_fill_large buys a large cap that opened 2.5-5% below its prior close once it turns up. On 48 large
caps' 1-minute bars it made +0.36R a trade since 2012 on mornings when 5%+ of the 48 gapped down 2.5%+
together, and lost 0.21R a trade on lone gap-downs. Buying every 2.5%+ gap-down at 09:31 and selling at
the close showed the same split. Before 2012 there was no difference, and the 48 are survivors picked on the
same data. This checks the idea on every S&P 500 stock, delisted ones included, on the Dev half.

No stops or sizing: an event table, like the other studies.

  breadth  the share of S&P 500 members (at the prior close, in the loaded half) that opened 2.5%+
           below their prior close; known once they have opened. A price-break day doesn't count,
           and days with fewer than --min-members (100) members are left out
  event    a stock that passed the universe at the prior close (--size sp500 by default: an S&P 500
           member) opens --min-gap (2.5) to --max-gap (5) % below its prior close, with no price
           break on the day
  trade    buy at the open, sell at that day's close. ret = after a --cost-pct (0.10) round trip;
           gross = before it; vs_spy = gross minus SPY's open-to-close that day
  SPY      SPY bought at the open and sold at the close, after --spy-cost-pct (0.02), by the same
           breadth, as the market's own part of the move
  pick     stated before the first run: the events on days with breadth 5%+ ("broad"), 2010 on.
           Worth building on if ret is clearly above 0 with month-clustered t above 2, positive in
           more than half the years and in the recent 5-year blocks, and above the lone days' ret

On the 48 large caps' minute bars, the first minute after the open lowered these trades (-0.09% for
2.5-5% gaps on broad days since 2012), so buying at the open, which the daily bars force, doesn't
flatter them even though breadth is only known a moment later.

Columns: days = distinct days, trades = stock-days, win_% = share of trades with ret > 0, t = month-
clustered t of ret (t_spy of vs_spy). Periods: before 2010 and 2010 on, side by side.

HOW TO RUN:
  python gapdown_breadth/breadth_events.py
  python gapdown_breadth/breadth_events.py --start 2010-01-01 --sample 10
  python gapdown_breadth/breadth_events.py --cost-pct 0.5             # the other event studies' round trip
  python gapdown_breadth/breadth_events.py --size all --min-gap 2.5 --max-gap 100
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

MIN_GAP, MAX_GAP = 2.5, 5.0         # % below the prior close, the event's band
BREADTH_GAP = 2.5                   # % down at the open that counts toward breadth
BROAD = 5.0                         # % of members gapping down: a "broad" day
MIN_MEMBERS = 100                   # members with an open and a prior close needed to read breadth
COST_PCT, SPY_COST_PCT = 0.10, 0.02  # % round trip: large caps, SPY
SPLIT_YEAR = 2010                   # the pick is judged from here on; earlier years are shown beside it
BIN_EDGES = [0, 5, 10, 20, 30, np.inf]
BIN_LABELS = ["under 5%", "5-10%", "10-20%", "20-30%", "30%+"]
GAP_BAND_EDGES = [3.5, 5, 10]      # gap-size split, % down, after --min-gap
PRICE_EDGES = [0, 20, 50, 100, np.inf]
INDEX_NAMES = ["S&P 500", "S&P 400", "S&P 600", "none"]
LOAD_PAD_DAYS = 10                  # calendar days loaded before --start for the prior close


def gaps(df, codes):
    """% open vs the same stock's prior close (NaN on its first row or a bad price), and whether
    the row is a price break (either way)."""
    o = df["open"].to_numpy(np.float64)
    prev_c = ce.shift(df["close"].to_numpy(np.float64), 1, codes)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where((o > 0) & (prev_c > 0), (o / prev_c - 1) * 100, np.nan)
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    return gap, brk


def breadth(df, codes, gap, brk, thr=BREADTH_GAP, min_members=MIN_MEMBERS):
    """One row per day: members = S&P 500 members at the prior close with a gap reading that
    day (no price break), down = how many opened thr% or more below the prior close, pct =
    down / members in %. Days with fewer than min_members are left out."""
    member = ce.shift((df["in_sp500"].fillna(0) > 0).to_numpy().astype(np.float64), 1, codes) > 0
    ok = member & np.isfinite(gap) & ~brk
    d = pd.DataFrame({"date": df["date"].to_numpy()[ok], "down": gap[ok] <= -thr})
    out = d.groupby("date")["down"].agg(members="size", down="sum")
    out = out[out["members"] >= min_members].copy()
    out["pct"] = out["down"] / out["members"] * 100
    return out


def bucket(pct):
    """Breadth % -> its bin label (NaN -> None)."""
    idx = np.digitize(np.asarray(pct, dtype=np.float64), BIN_EDGES[1:-1], right=False)
    lab = np.array(BIN_LABELS, dtype=object)[np.clip(idx, 0, len(BIN_LABELS) - 1)]
    return np.where(np.isfinite(np.asarray(pct, dtype=np.float64)), lab, None)


def spy_days(market, br, cost):
    """SPY open-to-close on each day with a breadth reading: gross and after cost, in %."""
    m = market.sort_values("date").set_index("date")
    if not {"spy_open", "spy_close"} <= set(m.columns):
        return pd.DataFrame(columns=["date", "gross", "ret", "breadth", "bin"])
    with np.errstate(invalid="ignore", divide="ignore"):
        oc = (m["spy_close"] / m["spy_open"].where(m["spy_open"] > 0) - 1) * 100
    s = pd.DataFrame({"gross": oc.reindex(br.index).to_numpy(), "breadth": br["pct"].to_numpy()},
                     index=pd.DatetimeIndex(br.index, name="date"))
    s["ret"] = s["gross"] - cost
    s = s.dropna(subset=["gross"])
    s["bin"] = bucket(s["breadth"])
    return s.reset_index()


def index_of(df, rows):
    """S&P index membership at the prior close of each row."""
    p = rows - 1
    idx = df[["in_sp500", "in_sp400", "in_sp600"]].iloc[p].fillna(0).to_numpy() > 0
    return np.select([idx[:, 0], idx[:, 1], idx[:, 2]], INDEX_NAMES[:3], INDEX_NAMES[3])


def events(df, codes, gap, brk, P, lo, hi, start=None, end=None):
    """Rows that open lo% to hi% below the prior close (lo included), no price break on the day,
    whose stock passed the universe (P, size included) at the prior close."""
    has_prev = ce.shift(np.ones(len(df)), 1, codes) > 0
    cand = np.flatnonzero(has_prev & np.isfinite(gap) & (gap <= -lo) & (gap > -hi) & ~brk)
    d = df["date"].to_numpy()
    if start:
        cand = cand[d[cand] >= np.datetime64(pd.Timestamp(start))]
    if end:
        cand = cand[d[cand] <= np.datetime64(pd.Timestamp(end))]
    if len(cand) == 0:
        return cand
    prev = np.unique(cand - 1)
    ok = pd.Series(rules.universe_mask(df.iloc[prev], P), index=prev)
    return cand[ok.reindex(cand - 1).fillna(False).to_numpy(dtype=bool)]


def day_trades(df, rows, gap, br, spy, cost):
    """Buy at the open, sell at the close, for each event row: a frame with the date, symbol,
    gap, gross / ret / vs_spy (%), the day's breadth and its bin."""
    o = df["open"].to_numpy(np.float64)[rows]
    c = df["close"].to_numpy(np.float64)[rows]
    dates = pd.DatetimeIndex(df["date"].to_numpy()[rows])
    with np.errstate(invalid="ignore", divide="ignore"):
        gross = np.where((o > 0) & (c > 0), (c / o - 1) * 100, np.nan)
    spy_oc = spy.set_index("date")["gross"].reindex(dates).to_numpy() if len(spy) else np.full(len(rows), np.nan)
    pct = br["pct"].reindex(dates).to_numpy()
    t = pd.DataFrame({"row": rows, "date": dates, "symbol": df["symbol"].iloc[rows].astype(str).to_numpy(),
                      "gap": gap[rows], "gross": gross, "ret": gross - cost, "vs_spy": gross - spy_oc,
                      "breadth": pct, "bin": bucket(pct)})
    return t.dropna(subset=["gross", "breadth"]).reset_index(drop=True)


def stats(g):
    out = {"days": g["date"].nunique(), "trades": len(g)}
    if len(g) == 0:
        return out
    out.update({"gross": g["gross"].mean(), "ret": g["ret"].mean(), "vs_spy": g["vs_spy"].mean(),
                "win_%": (g["ret"] > 0).mean() * 100, "t": ce.tstat_by_month(g["ret"], g["date"]),
                "t_spy": ce.tstat_by_month(g["vs_spy"], g["date"])})
    return out


def spy_stats(s):
    out = {"days": len(s)}
    if len(s):
        out.update({"gross": s["gross"].mean(), "ret": s["ret"].mean(), "win_%": (s["ret"] > 0).mean() * 100,
                    "t": ce.tstat_by_month(s["ret"], s["date"])})
    return out


def periods(g, split_year=SPLIT_YEAR):
    y = pd.DatetimeIndex(g["date"]).year
    return [(f"before {split_year}", g[y < split_year]), (f"{split_year} on", g[y >= split_year])]


def by_bin(t, fn, broad=BROAD, split_year=SPLIT_YEAR):
    """fn's stats for each breadth bin and for broad days (breadth >= broad), before and from
    split_year: one row per bin and period."""
    rows = []
    groups = [(lab, t["bin"] == lab) for lab in BIN_LABELS] + [(f"broad ({broad:g}%+)", t["breadth"] >= broad)]
    for lab, m in groups:
        for name, g in periods(t[m.to_numpy()], split_year):
            rows.append({"breadth": lab, "years": name, **fn(g)})
    return pd.DataFrame(rows)


def split_rows(groups):
    """stats() for each (group, frame, [(label, mask on frame)])."""
    rows = []
    for group, frame, buckets in groups:
        for label, m in buckets:
            rows.append({"group": group, "bucket": label, **stats(frame[np.asarray(m, dtype=bool)])})
    return pd.DataFrame(rows)


def edges_of(lo, inner):
    return [lo] + [x for x in inner if x > lo] + [np.inf]


def ranges(v, edges, fmt):
    """(label, mask) for each [lo, hi) of v."""
    v = np.asarray(v, dtype=np.float64)
    return [(f"{fmt(a)} to {fmt(b)}" if np.isfinite(b) else f"{fmt(a)}+", (v >= a) & (v < b))
            for a, b in zip(edges[:-1], edges[1:])]


def by_period(pick, lone, spy_b, key):
    """Broad-day trades, lone-day trades and SPY on broad days, by key (a function of the dates)."""
    rows = []
    keys = sorted(set(key(pick["date"])) | set(key(lone["date"])))
    for k in keys:
        b, ln = pick[key(pick["date"]) == k], lone[key(lone["date"]) == k]
        s = spy_b[key(spy_b["date"]) == k] if len(spy_b) else spy_b
        st_b, st_l = stats(b), stats(ln)
        rows.append({"period": k, "broad_days": st_b["days"], "broad_trades": st_b["trades"],
                     "broad_ret": st_b.get("ret", np.nan), "broad_vs_spy": st_b.get("vs_spy", np.nan),
                     "broad_t": st_b.get("t", np.nan), "lone_trades": st_l["trades"],
                     "lone_ret": st_l.get("ret", np.nan), "spy_broad_ret": s["ret"].mean() if len(s) else np.nan})
    return pd.DataFrame(rows)


def block(dates):
    y = pd.DatetimeIndex(dates).year // 5 * 5
    return np.asarray([f"{a}-{a + 4}" for a in y], dtype=object)


def year(dates):
    return np.asarray(pd.DatetimeIndex(dates).year)


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.set_defaults(size="sp500")
    parser.add_argument("--start", type=str, default=None, help="First day")
    parser.add_argument("--end", type=str, default=None, help="Last day")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Only once the rule is frozen.")
    parser.add_argument("--min-gap", type=float, default=MIN_GAP, help="Event: opens at least this %% down (2.5)")
    parser.add_argument("--max-gap", type=float, default=MAX_GAP, help="Event: and less than this %% down (5)")
    parser.add_argument("--broad", type=float, default=BROAD, help="Breadth %% that makes a broad day (5)")
    parser.add_argument("--min-members", type=int, default=MIN_MEMBERS,
                        help="S&P 500 members needed to read a day's breadth (100)")
    parser.add_argument("--cost-pct", type=float, default=COST_PCT, help="Round trip for a stock, %% (0.10)")
    parser.add_argument("--spy-cost-pct", type=float, default=SPY_COST_PCT, help="Round trip for SPY, %% (0.02)")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check on a chart")
    parser.add_argument("--save-trades", type=str, default=None, help="Write every event trade to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])

    t0 = time.time()
    cols = list(dict.fromkeys(rules.universe_columns(P) + ["open", "close", "bad_bar", "bad_bar_dn", "in_sp500",
                                                           "in_sp400", "in_sp600"]))
    load_from = (pd.Timestamp(args.start) - pd.Timedelta(days=LOAD_PAD_DAYS)) if args.start else None
    try:
        df = datastore.load_panel(cols, start=load_from, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python pipeline/build_data.py, then python pipeline/build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy() if hasattr(df["symbol"], "cat") else pd.factorize(df["symbol"])[0]
    market = pd.read_parquet(config.MARKET_FILE)
    spx200 = datastore.load_market(["spx_above200"]).set_index("date")["spx_above200"]

    gap, brk = gaps(df, codes)
    br = breadth(df, codes, gap, brk, BREADTH_GAP, args.min_members)
    if args.start:
        br = br[br.index >= pd.Timestamp(args.start)]
    if args.end:
        br = br[br.index <= pd.Timestamp(args.end)]
    spy = spy_days(market, br, args.spy_cost_pct)
    rows = events(df, codes, gap, brk, P, args.min_gap, args.max_gap, args.start, args.end)
    t = day_trades(df, rows, gap, br, spy, args.cost_pct)
    t["index"] = index_of(df, t["row"].to_numpy())
    every_p = dict(P, size="all")
    wide = day_trades(df, events(df, codes, gap, brk, every_p, args.min_gap, np.inf, args.start, args.end),
                      gap, br, spy, args.cost_pct)
    wide["index"] = index_of(df, wide["row"].to_numpy())

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("=" * 110)
    print(f"MARKET-WIDE GAP-DOWNS  events: {args.min_gap:g}-{args.max_gap:g}% gap-downs, size={P['size']}  "
          f"broad = {args.broad:g}%+ of S&P 500 members gapping {BREADTH_GAP:g}%+ down  universe={args.universe}  "
          f"start={args.start}  end={args.end}  cost {args.cost_pct:g}% (SPY {args.spy_cost_pct:g}%)  "
          f"({time.time() - t0:.1f}s)")
    print("=" * 110)
    if br.empty or t.empty:
        print("No days with a breadth reading, or no events.")
        return
    print(f"{len(br):,} days with a breadth reading ({br.index.min():%Y-%m-%d} to {br.index.max():%Y-%m-%d}), "
          f"median {int(br['members'].median())} members a day. Share of days by breadth: "
          + ", ".join(f"{lab} {(bucket(br['pct']) == lab).mean() * 100:.1f}%" for lab in BIN_LABELS) + ".")
    if spy.empty:
        print("No SPY open and close in the market file: the SPY columns are left out.")
    print(f"{len(t):,} events ({t['date'].nunique():,} days); {len(wide):,} gap-downs of {args.min_gap:g}%+ in any "
          f"stock that passed the universe (size all).")

    print(f"\nTHE EVENTS BY BREADTH: {args.min_gap:g}-{args.max_gap:g}% gap-downs ({P['size']}), bought at the open, "
          f"sold at the close.")
    print(by_bin(t, stats, args.broad).round(2).to_string(index=False))
    if len(spy):
        print(f"\nSPY bought at the open and sold at the close, by the same breadth (ret after {args.spy_cost_pct:g}%):")
        print(by_bin(spy, spy_stats, args.broad).round(3).to_string(index=False))
    print("\nEVERY STOCK THAT DAY: the S&P 500 members' average open-to-close by breadth, gap or not "
          "(their gross; ret after the stock round trip), the basket the events are measured against:")
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    member = ce.shift((df["in_sp500"].fillna(0) > 0).to_numpy().astype(np.float64), 1, codes) > 0
    keep = member & np.isfinite(gap) & ~brk & (c > 0)
    dd = pd.DatetimeIndex(df["date"].to_numpy()[keep])
    basket = pd.DataFrame({"date": dd, "gross": (c[keep] / o[keep] - 1) * 100})
    basket["ret"] = basket["gross"] - args.cost_pct
    basket["vs_spy"] = basket["gross"] - (spy.set_index("date")["gross"].reindex(dd).to_numpy() if len(spy)
                                          else np.nan)
    basket["breadth"] = br["pct"].reindex(dd).to_numpy()
    basket = basket.dropna(subset=["breadth"])
    basket["bin"] = bucket(basket["breadth"])
    print(by_bin(basket, stats, args.broad).round(2).to_string(index=False))

    since = t[pd.DatetimeIndex(t["date"]).year >= SPLIT_YEAR]
    pick = since[since["breadth"] >= args.broad]
    lone = since[since["breadth"] < args.broad]
    print(f"\nPICK: the events on broad days, {SPLIT_YEAR} on ({len(pick):,} trades on {pick['date'].nunique():,} days), "
          f"next to the lone days, split by what was known at the open. The gap-size and index rows take any "
          f"stock that passed the universe (size all).")
    wide_since = wide[pd.DatetimeIndex(wide["date"]).year >= SPLIT_YEAR]
    pct = lambda x: f"{x:g}%"
    money = lambda x: f"${x:g}"
    groups = []
    for name, src, ev in [("broad", wide_since[wide_since["breadth"] >= args.broad], pick),
                          ("lone", wide_since[wide_since["breadth"] < args.broad], lone)]:
        g = src["gap"].abs().to_numpy()
        prev = ev["row"].to_numpy() - 1
        up = spx200.reindex(pd.DatetimeIndex(df["date"].to_numpy()[prev])).to_numpy()
        price = df["unadj_close"].to_numpy(np.float64)[prev]
        groups += [
            (f"{name}: gap size", src, ranges(g, edges_of(args.min_gap, GAP_BAND_EDGES), pct)),
            (f"{name}: S&P index, {args.min_gap:g}-{args.max_gap:g}% gaps", src,
             [(k, (src["index"] == k).to_numpy() & (g < args.max_gap)) for k in INDEX_NAMES]),
            (f"{name}: S&P vs its 200-day", ev, [("above", up == 1), ("below", up == 0)]),
            (f"{name}: price", ev, ranges(price, PRICE_EDGES, money)),
        ]
    print(split_rows(groups).round(2).to_string(index=False))

    spy_since = spy[pd.DatetimeIndex(spy["date"]).year >= SPLIT_YEAR] if len(spy) else spy
    spy_b = spy_since[spy_since["breadth"] >= args.broad] if len(spy) else spy
    print("\nPICK BY 5-YEAR BLOCK (all years; lone = the events on the other days, spy_broad_ret = SPY on broad days):")
    blocks = by_period(t[t["breadth"] >= args.broad], t[t["breadth"] < args.broad],
                       spy[spy["breadth"] >= args.broad] if len(spy) else spy, block)
    print(blocks.round(2).to_string(index=False))
    yr = by_period(pick, lone, spy_b, year)
    print(f"\nPICK BY YEAR ({SPLIT_YEAR} on):")
    print(yr.round(2).to_string(index=False))
    y = yr["broad_ret"].dropna()
    st = stats(pick)
    print(f"Broad days: positive in {int((y > 0).sum())} of {len(y)} years. Lone days: "
          f"{stats(lone).get('ret', np.nan):+.2f}% a trade.")

    if args.sample and len(pick):
        s = pick.sample(min(args.sample, len(pick)), random_state=0).sort_values("date")
        print(f"\n{len(s)} random pick trades to check on a chart:")
        print(s[["symbol", "date", "gap", "gross", "breadth"]].round({"gap": 2, "gross": 2, "breadth": 2})
              .to_string(index=False))
    if args.save_trades:
        t.drop(columns=["row"]).to_csv(args.save_trades, index=False)
        print(f"\nSaved {len(t)} event trades to {args.save_trades}")

    ss = spy_stats(spy_b) if len(spy) else {"days": 0}
    print(f"\nThe pick, {SPLIT_YEAR} on: {st['trades']:,} trades, ret {st.get('ret', np.nan):+.3f}% a trade after "
          f"{args.cost_pct:g}% (t {st.get('t') or np.nan:.2f}), vs SPY {st.get('vs_spy', np.nan):+.3f}% "
          f"(t {st.get('t_spy') or np.nan:.2f}); lone days {stats(lone).get('ret', np.nan):+.3f}%. SPY on the broad "
          f"days: {ss['days']:,} days, {ss.get('ret', np.nan):+.3f}% a day after {args.spy_cost_pct:g}% "
          f"(t {ss.get('t') or np.nan:.2f}).")
    print("Worth building on, set before the first run: ret clearly above 0 with t above 2, positive in more than "
          "half the years and in the recent blocks, and above the lone days. The bin rows, the basket and the splits "
          "are context, not more picks.")

    if not args.no_log:
        res = {"pick": {k: (round(float(v), 4) if v is not None and np.isfinite(v) else None) for k, v in st.items()},
               "lone": {k: (round(float(v), 4) if v is not None and np.isfinite(v) else None)
                        for k, v in stats(lone).items()},
               "spy_broad": {k: (round(float(v), 4) if v is not None and np.isfinite(v) else None)
                             for k, v in spy_stats(spy_b).items()} if len(spy) else {}}
        research_log.append(
            question="Do large caps that gap down with the market rise from the open to the close?",
            phase="gapdown_breadth", split=args.universe, script="breadth_events.py",
            params={"start": args.start, "end": args.end, "min_gap": args.min_gap, "max_gap": args.max_gap,
                    "broad": args.broad, "min_members": args.min_members, "cost_pct": args.cost_pct,
                    "spy_cost_pct": args.spy_cost_pct, "size": P["size"], "liq_top_pct": P["liq_top_pct"],
                    "min_price": P["min_price"]},
            combinations=2, runs=1, results=res)


if __name__ == "__main__":
    main()
