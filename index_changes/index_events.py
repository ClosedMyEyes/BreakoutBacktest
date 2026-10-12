"""
S&P index changes — do stocks fade after they join an index and bounce after they leave?
=========================================================================================
Index funds must buy every stock added to the S&P 500 / 400 / 600 and sell every
stock that leaves, at the close before the change takes effect, whatever the
price. If that forced trading pushes prices too far, an added stock should lag
the S&P afterwards (short it) and a dropped one should beat it (buy it).
Passive money has grown a lot since 2010, so unlike the breakout and stretch
setups the reason for this edge should be stronger now, not weaker.

No setups, stops or exits: an event table, like the stretch study.

Events come from the daily index flags (in_sp500 / in_sp400 / in_sp600). A
stock's index is the one it is in that day (500, 400, 600 or none). An event is
the first day the data shows a different index, for example 600 -> 400 (a
promotion) or 600 -> none (a drop). A stock that leaves for "none" and stops
trading within 20 bars was bought out: counted, but left out of the tables.

  day 0   first day in the new index. S&P changes take effect before that day's
          open, so index funds trade at the close of day -1 (the "flow day").
          vol_-1 and vol_0 show the median volume on each day against the stock's
          usual volume: a big vol_-1 confirms the alignment.

Returns (all in %, close to close, stock minus the S&P over the same days):
  pre5     day -6 close to day -1 close: the run-up into the change. Not tradable
           from the data alone (the announcement comes about a week before; we only
           see the effective day), shown for context
  flow     day -2 close to day -1 close: the flow day itself
  postH    day -1 close (the flow-day close, the MOC price) to H days later: what a
           trade entered at the flow-day close earns. In real life you know the change
           from S&P's press release, so this is tradable
  late10   day 0 close to 10 days later: entering only once the data shows the change
  trade10  post10 in the natural direction (short the adds, buy the drops) minus the
           round-trip cost. Against the S&P, so it assumes the S&P leg is hedged
  t10      month-clustered t-stat of post10

The S&P 500 is a poor yardstick for a small cap, so each focus transition also
gets a detail section against the stock's own index: an equal-weight average of
the Dev stocks in that index each day (the new index for an add, the old one for
a drop to none). It shows every hold length, and splits post10 by the run-up
(pre5), the flow day, price and flow-day volume.

The days the S&P 400 (1991) and 600 (1994) were launched put hundreds of stocks
into an index at once. Any day with more than 60 events of one transition is
treated as a launch and left out.

Windows that span a price break are left out. Dividends aren't charged.

HOW TO RUN:
  python index_changes/index_events.py                          # every year, Dev stocks
  python index_changes/index_events.py --start 2010-01-01       # the years that broke the other setups
  python index_changes/index_events.py --focus "600->none"      # detail and by-year tables for one transition
  python index_changes/index_events.py --start 2010-01-01 --focus "none->600,600->none"
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Kept in its own folder (a separate idea from the spike short); shares the
# repo's data loading, so the repo root goes on the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import datastore
import research_log

STATE_NAMES = {3: "500", 2: "400", 1: "600", 0: "none"}
POST_H = [1, 5, 10, 20]
MAIN_H = 10
STABLE_BARS = 5       # the old index must have held this many bars before the change
BUYOUT_BARS = 20      # left for "none" and stopped trading within this many bars = buyout
VOL_BASE = (-25, -6)  # rows used for the stock's usual volume
LAUNCH_BATCH = 60     # more events than this of one transition on one day = an index launch


def index_state(df):
    """500 / 400 / 600 / none as 3 / 2 / 1 / 0 (a stock is only ever in one)."""
    s = np.zeros(len(df), dtype=np.int8)
    s[df["in_sp600"].to_numpy() > 0] = 1
    s[df["in_sp400"].to_numpy() > 0] = 2
    s[df["in_sp500"].to_numpy() > 0] = 3
    return s


def find_events(df, codes):
    """Rows (sorted by symbol, date) where a stock's index changes, after holding
    the old one for STABLE_BARS bars. Returns a DataFrame of events with the row,
    the old and new index, and whether it looks like a buyout."""
    n = len(df)
    st = index_state(df)
    idx = np.arange(n)
    first = pd.Series(idx).groupby(codes).transform("min").to_numpy()
    last = pd.Series(idx).groupby(codes).transform("max").to_numpy()
    rows = np.flatnonzero((idx - first >= STABLE_BARS) & (st != np.roll(st, 1)))
    stable = np.ones(len(rows), dtype=bool)
    for k in range(2, STABLE_BARS + 1):
        stable &= st[rows - k] == st[rows - 1]
    rows = rows[stable]
    ev = pd.DataFrame({
        "row": rows,
        "symbol": df["symbol"].to_numpy()[rows],
        "date": df["date"].to_numpy()[rows],
        "old": st[rows - 1], "new": st[rows],
    })
    ev["transition"] = [f"{STATE_NAMES[a]}->{STATE_NAMES[b]}" for a, b in zip(ev["old"], ev["new"])]
    ev["kind"] = np.where(ev["new"] > ev["old"], "add", "drop")
    ev["buyout"] = (ev["new"] == 0) & (last[rows] - rows < BUYOUT_BARS)
    ev["first"], ev["last"] = first[rows], last[rows]
    return ev


def member_levels(df, codes):
    """Equal-weight index level of each S&P index's Dev members, by date: the
    average daily return of the stocks that were in it at the prior close
    (days with a price break left out), chained. Columns 1 / 2 / 3 = 600 / 400 / 500."""
    st = index_state(df)
    c = df["close"].to_numpy(np.float64)
    same = np.r_[False, codes[1:] == codes[:-1]]
    prev_c = np.r_[np.nan, c[:-1]]
    prev_st = np.r_[0, st[:-1]]
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    ok = same & ~brk & (prev_st > 0) & (prev_c > 0)
    d = pd.DataFrame({"date": df["date"].to_numpy()[ok], "state": prev_st[ok],
                      "ret": c[ok] / prev_c[ok] - 1})
    daily = d.groupby(["date", "state"])["ret"].mean().unstack("state")
    all_dates = pd.Index(np.sort(df["date"].unique()), name="date")
    daily = daily.reindex(index=all_dates, columns=[1, 2, 3]).fillna(0.0)
    return (1 + daily).cumprod()


def drop_launches(ev):
    """Leave out index launch days: more than LAUNCH_BATCH events of one transition
    on one day (the S&P 400 in 1991, the 600 in 1994). Returns (events, number dropped)."""
    batch = ev.groupby(["date", "transition"])["row"].transform("size")
    launch = (batch > LAUNCH_BATCH).to_numpy()
    return ev[~launch].reset_index(drop=True), int(launch.sum())


def add_windows(ev, df, spx, levels=None):
    """Stock-minus-S&P returns for each window, by offsets from the event row, and
    (with levels) stock minus its own index's members as {name}_idx."""
    c = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    cb = np.cumsum(brk)
    dates = df["date"].to_numpy()
    spx_c = spx.set_index("date")["spx_close"]
    r, first, last = ev["row"].to_numpy(), ev["first"].to_numpy(), ev["last"].to_numpy()
    own = np.where(ev["new"].to_numpy() > 0, ev["new"].to_numpy(), ev["old"].to_numpy())

    def window(a, b, name):
        i, j = r + a, np.minimum(r + b, last)        # a stock that ends early uses its last close
        ok = (i >= first) & (j > i)
        i, j = np.where(ok, i, r), np.where(ok, j, r)
        raw = c[j] / c[i] - 1
        sp = spx_c.reindex(dates[j]).to_numpy() / spx_c.reindex(dates[i]).to_numpy() - 1
        bad = ~ok | (cb[j] - cb[i] > 0)
        raw[bad] = np.nan
        ev[f"raw_{name}"] = raw * 100
        ev[name] = (raw - sp) * 100
        if levels is not None:
            lv = levels.to_numpy()
            pos = levels.index.get_indexer(pd.DatetimeIndex(dates))
            col = np.clip(own - 1, 0, 2)
            bi, bj = pos[i], pos[j]
            ix = lv[bj, col] / lv[bi, col] - 1
            ix[(bi < 0) | (bj < 0)] = np.nan
            ev[f"{name}_idx"] = (raw - ix) * 100

    window(-6, -1, "pre5")
    window(-2, -1, "flow")
    for h in POST_H:
        window(-1, -1 + h, f"post{h}")
    window(0, MAIN_H, "late10")

    lo, hi = r + VOL_BASE[0], r + VOL_BASE[1]
    ok = lo >= first
    base = np.full(len(r), np.nan)
    for k in np.flatnonzero(ok):
        base[k] = np.nanmean(vol[lo[k]:hi[k] + 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["vol_-1"] = vol[np.maximum(r - 1, 0)] / base
        ev["vol_0"] = vol[r] / base
    return ev


def tstat_by_month(x, dates):
    """t-stat of the mean with events grouped by month (changes come in batches
    at each quarterly rebalance, so same-month events aren't independent)."""
    x = pd.Series(np.asarray(x, dtype=np.float64), index=pd.to_datetime(np.asarray(dates)))
    x = x.dropna()
    if len(x) < 10:
        return None
    e = x - x.mean()
    g = e.groupby(x.index.to_period("M")).sum()
    if len(g) < 5:
        return None
    se = np.sqrt((g ** 2).sum() * len(g) / (len(g) - 1)) / len(x)
    return float(x.mean() / se) if se > 0 else None


def round_trip_cost():
    """% of price: slippage both sides plus commission at about a $40 stock."""
    return 2 * config.DEFAULT_SLIPPAGE_PCT + 2 * config.COMMISSION_PER_SHARE / 40 * 100


def stats(g, years):
    """One table row for a group of events."""
    out = {"events": len(g), "per_yr": len(g) / years}
    if len(g) == 0:
        return out
    for col in ["pre5", "flow"] + [f"post{h}" for h in POST_H] + ["late10"]:
        out[col] = g[col].mean()
    m = f"post{MAIN_H}"
    out[f"raw{MAIN_H}"] = g[f"raw_{m}"].mean()
    out[f"med{MAIN_H}"] = g[m].median()
    out[f"up{MAIN_H}_%"] = (g[m].dropna() > 0).mean() * 100
    out[f"t{MAIN_H}"] = tstat_by_month(g[m], g["date"])
    sign = np.where(g["kind"] == "add", -1.0, 1.0)
    out[f"trade{MAIN_H}"] = np.nanmean(sign * g[m].to_numpy()) - round_trip_cost()
    out["vol_-1"] = g["vol_-1"].median()
    out["vol_0"] = g["vol_0"].median()
    return out


def span_years(dates):
    d = pd.to_datetime(pd.Series(dates))
    return max(0.1, (d.max() - d.min()).days / 365.25) if len(d) else 0.1


def group_table(ev, key, years):
    rows = [{key: k, **stats(g, years)} for k, g in ev.groupby(key, sort=False)]
    return pd.DataFrame(rows)


def detail(g, years):
    """Hold lengths and splits for one transition, against its own index."""
    sign = np.where(g["kind"] == "add", -1.0, 1.0)
    cost = round_trip_cost()
    rows = []
    for h in POST_H:
        x = g[f"post{h}_idx"]
        rows.append({"hold": h, "vs_spx": g[f"post{h}"].mean(), "vs_index": x.mean(),
                     "med_vs_index": x.median(), "t_vs_index": tstat_by_month(x, g["date"]),
                     "trade_vs_index": np.nanmean(sign * x.to_numpy()) - cost,
                     "trade_%_per_day": (np.nanmean(sign * x.to_numpy()) - cost) / h})
    print(f"  by hold (entry at the flow-day close; trade = in the natural direction minus {cost:.2f}%):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    m = f"post{MAIN_H}_idx"

    def split_row(name, mask):
        s = g.loc[mask]
        x = s[m].to_numpy()
        sg = np.where(s["kind"] == "add", -1.0, 1.0)
        return {"split": name, "events": len(s), "per_yr": len(s) / years,
                f"post{MAIN_H}_vs_spx": s[f"post{MAIN_H}"].mean(), f"post{MAIN_H}_vs_index": np.nanmean(x),
                f"t{MAIN_H}": tstat_by_month(s[m], s["date"]),
                f"trade{MAIN_H}": np.nanmean(sg * x) - cost}

    rows = [split_row("all", np.ones(len(g), dtype=bool))]
    for col, label in [("pre5", "run-up pre5"), ("vol_-1", "flow-day volume")]:
        v = g[col]
        lo, hi = v.quantile(1 / 3), v.quantile(2 / 3)
        rows += [split_row(f"{label} low third (< {lo:.1f})", (v < lo).to_numpy()),
                 split_row(f"{label} middle third", ((v >= lo) & (v < hi)).to_numpy()),
                 split_row(f"{label} high third (>= {hi:.1f})", (v >= hi).to_numpy())]
    rows += [split_row("flow day up vs S&P", (g["flow"] > 0).to_numpy()),
             split_row("flow day down vs S&P", (g["flow"] <= 0).to_numpy())]
    px = g["price"]
    rows += [split_row("price $5-15", (px < 15).to_numpy()),
             split_row("price $15-40", ((px >= 15) & (px < 40)).to_numpy()),
             split_row("price $40+", (px >= 40).to_numpy())]
    print(f"\n  post{MAIN_H} split by what was known at the flow-day close (vs its own index):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def block_of(dates):
    y = pd.to_datetime(pd.Series(dates)).dt.year // 5 * 5
    return (y.astype(str) + "-" + (y + 4).astype(str)).to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="First event date")
    parser.add_argument("--end", type=str, default=None, help="Last event date")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Only once the rule is frozen.")
    parser.add_argument("--min-price", type=float, default=5.0,
                        help="Printed price at the flow-day close must be at least this")
    parser.add_argument("--focus", type=str, default=None,
                        help='A transition such as "600->none" for the by-year table '
                             "(default: the biggest add and the biggest drop)")
    parser.add_argument("--save-events", type=str, default=None, help="Write every event to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()

    t0 = time.time()
    cols = ["close", "unadj_close", "volume", "bad_bar", "bad_bar_dn", "in_sp500", "in_sp400", "in_sp600"]
    try:
        # Full history: the index a stock was in before --start still matters
        df = datastore.load_panel(cols, universe=args.universe, unlock_holdout=args.unlock_holdout)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy()
    spx = pd.read_parquet(config.MARKET_FILE, columns=["date", "spx_close"])

    ev = find_events(df, codes)
    ev, n_launch = drop_launches(ev)
    if args.start:
        ev = ev[ev["date"] >= pd.Timestamp(args.start)]
    if args.end:
        ev = ev[ev["date"] <= pd.Timestamp(args.end)]
    ev = ev.reset_index(drop=True)
    n_buyout = int(ev["buyout"].sum())
    ev = ev[~ev["buyout"]].reset_index(drop=True)
    px = df["unadj_close"].to_numpy(np.float64)[np.maximum(ev["row"].to_numpy() - 1, 0)]
    n_cheap = int((px < args.min_price).sum())
    ev["price"] = px
    ev = ev[~(px < args.min_price)].reset_index(drop=True)
    ev = add_windows(ev, df, spx, member_levels(df, codes))

    print("=" * 100)
    print(f"S&P INDEX CHANGES  universe={args.universe}  start={args.start}  end={args.end}  "
          f"min price ${args.min_price:g}  ({time.time() - t0:.1f}s)")
    if ev.empty:
        print("No index changes found. The index flags (in_sp500/400/600) never change in this data "
              "(the synthetic data keeps them fixed).")
        return
    years = span_years(ev["date"])
    print(f"{len(ev):,} events, {years:.1f} years ({ev['date'].min():%Y-%m-%d} to "
          f"{ev['date'].max():%Y-%m-%d}). Left out: {n_buyout} buyouts (left the index and stopped "
          f"trading within {BUYOUT_BARS} bars), {n_cheap} under ${args.min_price:g}, "
          f"{n_launch} on index launch days (all years).")
    print("=" * 100)
    pd.set_option("display.width", 250)

    print(f"\nBy transition. All returns are % vs the S&P. Adds should have post < 0 (short them), drops "
          f"post > 0 (buy them). trade{MAIN_H} = post{MAIN_H} in that direction minus a "
          f"{round_trip_cost():.2f}% round trip.")
    print(f"vol_-1 / vol_0 = median volume on the flow day / day 0 vs usual (big vol_-1 = alignment right).")
    t = group_table(ev, "transition", years).sort_values("events", ascending=False)
    print(t.round(2).to_string(index=False))

    print(f"\nAll adds vs all drops:")
    print(group_table(ev, "kind", years).round(2).to_string(index=False))

    ev["block"] = block_of(ev["date"])
    print(f"\nBy 5-year block (does the edge grow with passive money?):")
    for kind in ["add", "drop"]:
        g = ev[ev["kind"] == kind]
        rows = []
        for b, gb in g.groupby("block"):
            rows.append({"block": b, **stats(gb, span_years(gb["date"]))})
        print(f"\n  {kind}s")
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.focus:
        focus = [f.strip() for f in args.focus.split(",") if f.strip()]
    else:
        counts = ev.groupby(["kind", "transition"]).size().reset_index(name="n")
        focus = [counts[counts["kind"] == k].sort_values("n").iloc[-1]["transition"]
                 for k in ["add", "drop"] if (counts["kind"] == k).any()]
    for f in focus:
        g = ev[ev["transition"] == f]
        if g.empty:
            print(f"\nNo {f} events. Transitions found: {sorted(ev['transition'].unique())}")
            continue
        print(f"\n{f} IN DETAIL ({len(g)} events, {len(g) / years:.1f} a year):")
        detail(g, years)
        rows = []
        for y, gy in g.groupby(g["date"].dt.year):
            s = stats(gy, 1.0)
            sg = np.where(gy["kind"] == "add", -1.0, 1.0)
            rows.append({"year": y, **{k: s.get(k) for k in
                                       ["events", "pre5", "flow", "post5", f"post{MAIN_H}", "post20",
                                        f"trade{MAIN_H}"]},
                         f"post{MAIN_H}_idx": gy[f"post{MAIN_H}_idx"].mean(),
                         f"trade{MAIN_H}_idx": np.nanmean(sg * gy[f"post{MAIN_H}_idx"].to_numpy())
                         - round_trip_cost()})
        print(f"\n{f} by year (_idx = against its own index):")
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.save_events:
        keep = ["symbol", "date", "transition", "kind", "price", "pre5", "flow"] + \
               [f"post{h}" for h in POST_H] + [f"post{h}_idx" for h in POST_H] + \
               ["late10", "vol_-1", "vol_0"]
        ev[keep].to_csv(args.save_events, index=False)
        print(f"\nSaved {len(ev)} events to {args.save_events}")

    print(f"\nWorth building on = trade{MAIN_H} clearly above 0 with t{MAIN_H} above 2 for a transition "
          f"with a few dozen events a year, and positive in the 2010-2014 / 2015-2019 / 2020-2024 blocks.")

    if not args.no_log:
        summary = group_table(ev, "kind", years)
        res = {r["kind"]: {k: (round(float(v), 3) if v is not None and np.isfinite(v) else None)
                           for k, v in r.items() if k != "kind"} for _, r in summary.iterrows()}
        research_log.append(
            question="Do stocks fade after joining an S&P index and bounce after leaving?",
            phase="index", split=args.universe, script="index_events.py",
            params={"start": args.start, "end": args.end, "min_price": args.min_price,
                    "focus": args.focus},
            combinations=1, runs=1, results=res)


if __name__ == "__main__":
    main()
