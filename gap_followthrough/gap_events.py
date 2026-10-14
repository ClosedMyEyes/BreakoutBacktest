"""
Big-gap follow-through — do liquid stocks keep going after a big gap on heavy volume?
=====================================================================================
Idea #4 in ideas/next_setups.md. A liquid stock that opens 5%+ away from the prior
close on 3x+ its usual volume has nearly always had news, usually earnings. Prices
drifting on in the direction of an earnings surprise for weeks afterwards is one of
the oldest documented edges, but it is reported to have shrunk in large caps since
2010. This checks whether going with the gap is still worth trading after costs.

No setups, stops or sizing: an event table, like the lock-up and index studies.

  event   day 0 opens --min-gap % (5) or more above the prior close (an up gap) or below
          it (a down gap), on --min-vol-x (3) times or more the stock's average volume
          over the 50 days before. The stock passed the universe at the prior close
          ($5, top 50% by 50-day dollar volume, no price break in the past year). Only a
          stock's first such gap in --dedupe (20) bars counts, and a gap day with a price
          break is left out
  trade   go with the gap: buy an up gap, short a down gap. In at the day 0 close (the
          volume is only known then) or the next open, out at the close of day H. A
          stock that stops trading is sold at its last close; a window across a price
          break in the trade's favour (up for a buy, down for a short) is left out
  pick    stated before the first run: in at the day 0 close, out at the close of day 20,
          each direction on its own (--pick close:20)

Columns (in %, after a round trip and, for shorts, --borrow-pct a year for the days held):
  ret       what the trade makes
  hedged    the same against the S&P (a buy hedged with an S&P short, a short with an
            S&P buy)
  med / win_% / worst5%   median, share of winners, the 1-in-20 worst trade
  adv20_%   share that went 20%+ against the trade at some point before the exit
  t / t_hedged            month-clustered t of ret and hedged

HOW TO RUN:
  python gap_followthrough/gap_events.py --start 2010-01-01
  python gap_followthrough/gap_events.py --start 2010-01-01 --pick open:20 --sample 10
  python gap_followthrough/gap_events.py --start 2010-01-01 --min-gap 10 --min-vol-x 5
  python gap_followthrough/gap_events.py --start 2010-01-01 --size sp500        # S&P 500 stocks only

The day-by-day profile shows the move in the gap's direction around day 0 against
the S&P. The pick's table splits both directions by gap size, volume, S&P index,
price, dollar volume, what the stock did after the open on day 0, its 60-day move
before the gap, and the stock and the S&P against their 200-day averages.
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

MIN_GAP = 5.0                       # % open vs prior close
MIN_VOL_X = 3.0                     # day 0 volume vs the 50 days before
VOL_BARS, VOL_MIN_BARS = 50, 20     # the stock's usual volume, and the least history for it
DEDUPE = 20                         # a stock's first gap in this many bars
HOLDS = [1, 3, 5, 10, 20, 40, 60]   # exit at the close of day H
ENTRIES = ["close", "open"]         # day 0 close, or day 1 open
PICK = "close:20"
PROFILE = range(-5, 21)
RUN_BARS = 60                       # the move before the gap
ADVERSE = 20.0                      # % against the trade counted in adv20_%
LOAD_PAD_DAYS = 120                 # calendar days loaded before --start for the 50- and 60-bar look-backs
GAP_BINS = [10, 20]                 # |gap| %, after --min-gap
VOLX_BINS = [5, 10]                 # volume x, after --min-vol-x
PRICE_BINS = [0, 10, 20, 50, np.inf]
DV_BINS = [0, 5, 20, 100, np.inf]   # $M a day
RUN_EDGES = [-np.inf, -20, 0, 20, np.inf]
HOLD_BAND = 2.0                     # day 0 close within this % of the open = held near the open
SIDES = [("up", 1), ("down", -1)]


def parse_pick(spec):
    """'close:20' -> ('close', 20)."""
    entry, hold = str(spec).split(":")
    if entry not in ENTRIES or int(hold) < 1:
        raise ValueError(spec)
    return entry, int(hold)


def stock_bounds(codes):
    """First and last row of each row's stock (rows sorted by stock, then date)."""
    n = len(codes)
    idx = np.arange(n)
    new = np.r_[True, codes[1:] != codes[:-1]]
    end = np.r_[codes[1:] != codes[:-1], True]
    first = np.maximum.accumulate(np.where(new, idx, 0))
    last = np.minimum.accumulate(np.where(end, idx, n)[::-1])[::-1]
    return first, last


def usual_volume(vol, first, bars=VOL_BARS, min_bars=VOL_MIN_BARS):
    """Average volume over the `bars` bars before each row (same stock), NaN with fewer
    than min_bars of them."""
    v = np.asarray(vol, dtype=np.float64)
    fin = np.isfinite(v)
    cs = np.r_[0.0, np.cumsum(np.where(fin, v, 0.0))]
    cn = np.r_[0, np.cumsum(fin)]
    idx = np.arange(len(v))
    lo = np.maximum(idx - bars, first)
    s, n = cs[idx] - cs[lo], cn[idx] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_bars, s / n, np.nan)


def find_gaps(df, codes, min_gap=MIN_GAP, min_vol_x=MIN_VOL_X, dedupe=DEDUPE):
    """Rows that gap min_gap% or more on min_vol_x times the usual volume, the first such
    gap of the stock in `dedupe` bars, with no price break on the day. Returns a frame
    with row / first / last / sign (+1 up, -1 down) / gap / vol_x, and the counts."""
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    first, last = stock_bounds(codes)
    idx = np.arange(len(df))
    has_prev = idx > first
    prev_c = np.r_[np.nan, c[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(has_prev & (prev_c > 0), (o / prev_c - 1) * 100, np.nan)
        vol_x = df["volume"].to_numpy(np.float64) / usual_volume(df["volume"].to_numpy(), first)
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    big = (np.abs(gap) >= min_gap) & (vol_x >= min_vol_x)
    cand = np.flatnonzero(big)
    prev_cand = np.r_[-10 ** 9, cand[:-1]]
    repeat = (prev_cand >= first[cand]) & (cand - prev_cand <= dedupe)
    counts = {"gaps with the volume": int(len(cand)), "repeat within the dedupe bars": int(repeat.sum()),
              "price break on the day": int((brk[cand] & ~repeat).sum())}
    keep = cand[~repeat & ~brk[cand]]
    ev = pd.DataFrame({"row": keep, "first": first[keep], "last": last[keep],
                       "sign": np.where(gap[keep] > 0, 1, -1), "gap": gap[keep], "vol_x": vol_x[keep]})
    return ev, counts


def select(df, codes, P, args, keep_market=None):
    """Gaps inside the dates whose stock passed the universe at the prior close (and the
    market filter on day 0). Returns (events, counts)."""
    ev, counts = find_gaps(df, codes, args.min_gap, args.min_vol_x, args.dedupe)
    d = df["date"].to_numpy()
    ev["date"] = d[ev["row"].to_numpy()]
    if args.start:
        ev = ev[ev["date"] >= pd.Timestamp(args.start)]
    if args.end:
        ev = ev[ev["date"] <= pd.Timestamp(args.end)]
    ev = ev.reset_index(drop=True)
    counts["inside the dates"] = len(ev)
    prev = ev["row"].to_numpy() - 1
    sub = df.iloc[np.unique(prev)] if len(prev) else df.iloc[:0]
    ok_rows = pd.Series(rules.universe_mask(sub, P), index=sub.index)
    ok = ok_rows.reindex(prev).fillna(False).to_numpy(dtype=bool)
    if keep_market is not None:
        ok &= keep_market[ev["row"].to_numpy()]
    ev = ev[ok].reset_index(drop=True)
    ev["symbol"] = df["symbol"].astype(str).to_numpy()[ev["row"].to_numpy()]
    return ev, counts


def trades(ev, df, spx, entry, hold, borrow_pct):
    """Go with the gap: in at the day 0 close or the day 1 open, out at the close of day
    hold. Returns ret_ / hedged_ / adv_ columns named entry:hold (ret and hedged in %)."""
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cup = np.cumsum((df["bad_bar"].fillna(0) > 0).to_numpy())
    cdn = np.cumsum((df["bad_bar_dn"].fillna(0) > 0).to_numpy())
    dates = df["date"].to_numpy()
    r, last, sign = ev["row"].to_numpy(), ev["last"].to_numpy(), ev["sign"].to_numpy()
    j = np.minimum(r + hold, last)
    ok = j > r
    if entry == "open":
        ok &= r + 1 <= last
        px = np.where(ok, o[np.minimum(r + 1, last)], np.nan)
    else:
        px = c[r]
    ok &= px > 0
    raw = c[j] / px - 1
    brk_up, brk_dn = cup[j] - cup[r] > 0, cdn[j] - cdn[r] > 0
    bad = ~ok | np.where(sign > 0, brk_up, brk_dn)
    pos = spx.index.get_indexer(pd.DatetimeIndex(dates))
    sc = spx.to_numpy()
    pi, pj = pos[r], pos[j]
    sp = np.where((pi >= 0) & (pj >= 0), sc[np.maximum(pj, 0)] / sc[np.maximum(pi, 0)] - 1, np.nan)
    days = np.maximum(j - r, 1)
    cost = ce.round_trip_cost() + np.where(sign < 0, borrow_pct * days / 252, 0.0)
    worst = np.zeros(len(r))                       # furthest against the trade, % of the entry
    for k in range(1, hold + 1):
        jj = np.minimum(r + k, last)
        live = r + k <= j
        against = np.where(sign > 0, 1 - lo[jj] / px, h[jj] / px - 1) * 100
        worst = np.where(live, np.fmax(worst, against), worst)
    name = f"{entry}:{hold}"
    return {f"ret_{name}": np.where(bad, np.nan, sign * raw * 100 - cost),
            f"hedged_{name}": np.where(bad, np.nan, sign * (raw - sp) * 100 - cost),
            f"adv_{name}": np.where(bad, np.nan, (worst >= ADVERSE).astype(float))}


def stats(g, name, years):
    out = {"events": len(g), "per_yr": len(g) / years}
    s = g[f"ret_{name}"].dropna() if len(g) else pd.Series(dtype=float)
    if s.empty:
        return out
    out["ret"] = s.mean()
    out["hedged"] = g[f"hedged_{name}"].mean()
    out["med"] = s.median()
    out["win_%"] = (s > 0).mean() * 100
    out["worst5%"] = s.quantile(0.05)
    out["adv20_%"] = g[f"adv_{name}"].mean() * 100
    out["t"] = ce.tstat_by_month(g[f"ret_{name}"], g["date"])
    out["t_hedged"] = ce.tstat_by_month(g[f"hedged_{name}"], g["date"])
    return out


def profile(ev, df, spx, sign):
    """Average daily move in the gap's direction against the S&P around day 0, its running
    total from the day 0 close, and the median volume against the usual (one direction)."""
    g = ev[ev["sign"] == sign]
    c = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    pos = spx.index.get_indexer(pd.DatetimeIndex(df["date"].to_numpy()))
    sc = spx.to_numpy()
    r, first, last = g["row"].to_numpy(), g["first"].to_numpy(), g["last"].to_numpy()
    usual = g["usual_vol"].to_numpy()
    rows, run = [], 0.0
    for d in PROFILE:
        j = r + d
        ok = (j - 1 >= first) & (j <= last)
        jj, ii = np.where(ok, j, r), np.where(ok, j - 1, r)
        pi, pj = pos[ii], pos[jj]
        sp = np.where((pi >= 0) & (pj >= 0), sc[np.maximum(pj, 0)] / sc[np.maximum(pi, 0)] - 1, np.nan)
        x = np.where(ok & ~brk[jj], sign * (c[jj] / c[ii] - 1 - sp) * 100, np.nan)
        avg = np.nanmean(x) if np.isfinite(x).any() else np.nan
        if d > 0 and np.isfinite(avg):
            run += avg
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = np.where(ok, vol[jj] / usual, np.nan)
        rows.append({"day": d, "vs_spx": avg, "running": run if d > 0 else np.nan,
                     "with_%": (x[np.isfinite(x)] > 0).mean() * 100 if np.isfinite(x).any() else np.nan,
                     "med_vol_x": np.nanmedian(vx) if np.isfinite(vx).any() else np.nan})
    return pd.DataFrame(rows)


def features(ev, df, spx_above200):
    """What is known at the day 0 close (and the prior close), for the split table."""
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    r, first, sign = ev["row"].to_numpy(), ev["first"].to_numpy(), ev["sign"].to_numpy()
    p = r - 1
    ev["price"] = df["unadj_close"].to_numpy(np.float64)[p]
    ev["dv_m"] = df["dv50"].to_numpy(np.float64)[p] / 1e6
    idx = df[["in_sp500", "in_sp400", "in_sp600"]].to_numpy()[p] > 0
    ev["index"] = np.select([idx[:, 0], idx[:, 1], idx[:, 2]], ["S&P 500", "S&P 400", "S&P 600"], "none")
    with np.errstate(invalid="ignore", divide="ignore"):
        after_open = sign * (c[r] / o[r] - 1) * 100
        day0 = sign * (c[r] / c[p] - 1) * 100
        run = np.where(p - RUN_BARS >= first, sign * (c[p] / c[np.maximum(p - RUN_BARS, 0)] - 1) * 100, np.nan)
    ev["after_open"], ev["day0"], ev["run60"] = after_open, day0, run
    sma = df["sma200"].to_numpy(np.float64)[p]
    ev["above200"] = np.where(np.isfinite(sma), (c[p] > sma).astype(float), np.nan)
    ev["spx_above200"] = spx_above200.reindex(pd.DatetimeIndex(df["date"].to_numpy()[p])).to_numpy()
    return ev


def edges_of(lo, inner):
    return [lo] + [x for x in inner if x > lo] + [np.inf]


def bins_of(v, edges, fmt):
    """(label, mask) for each bin [lo, hi) of v."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if not np.isfinite(lo) or (lo == 0 and lo == edges[0]):
            label = f"under {fmt(hi)}"
        elif not np.isfinite(hi):
            label = f"{fmt(lo)}+"
        else:
            label = f"{fmt(lo)} to {fmt(hi)}"
        out.append((label, ((v >= lo) & (v < hi)).to_numpy()))
    return out


def split_table(g, name, years, min_gap, min_vol_x):
    """The pick for up gaps (buys) and down gaps (shorts) side by side, by what was
    known at the day 0 close."""
    def side(mask, sign):
        s = g.loc[np.asarray(mask, dtype=bool) & (g["sign"] == sign).to_numpy()]
        st = stats(s, name, years)
        return {"ev": st["events"], "ret": st.get("ret", np.nan), "hedged": st.get("hedged", np.nan),
                "win_%": st.get("win_%", np.nan), "t_h": st.get("t_hedged", np.nan)}

    pct = lambda x: f"{x:g}%"
    money = lambda x: f"${x:g}"
    held = g["after_open"]
    groups = [
        ("all", [("", np.ones(len(g), dtype=bool))]),
        ("gap size", bins_of(g["gap"].abs(), edges_of(min_gap, GAP_BINS), pct)),
        ("volume vs usual", bins_of(g["vol_x"], edges_of(min_vol_x, VOLX_BINS), lambda x: f"{x:g}x")),
        ("S&P index", [(k, (g["index"] == k).to_numpy()) for k in ["S&P 500", "S&P 400", "S&P 600", "none"]]),
        ("price", bins_of(g["price"], PRICE_BINS, money)),
        ("$ volume, $M/day", bins_of(g["dv_m"], DV_BINS, money)),
        ("day 0 after the open", [
            ("kept going 2%+", (held >= HOLD_BAND).to_numpy() & (g["day0"] > 0).to_numpy()),
            ("held near the open", (held.abs() < HOLD_BAND).to_numpy() & (g["day0"] > 0).to_numpy()),
            ("gave back 2%+", (held <= -HOLD_BAND).to_numpy() & (g["day0"] > 0).to_numpy()),
            ("filled the gap", (g["day0"] <= 0).to_numpy())]),
        ("60-day move before, in the gap's direction", bins_of(g["run60"], RUN_EDGES, pct)),
        ("stock vs its 200-day", [("above", (g["above200"] == 1).to_numpy()),
                                  ("below", (g["above200"] == 0).to_numpy())]),
        ("S&P vs its 200-day", [("above", (g["spx_above200"] == 1).to_numpy()),
                                ("below", (g["spx_above200"] == 0).to_numpy())]),
    ]
    rows = []
    for group, buckets in groups:
        for label, mask in buckets:
            up, dn = side(mask, 1), side(mask, -1)
            rows.append({"group": group, "bucket": label, **{f"up_{k}": v for k, v in up.items()},
                         **{f"dn_{k}": v for k, v in dn.items()}})
    return pd.DataFrame(rows)


def by_period(g, name, key):
    """The pick for each direction by 5-year block or year (up_ = buys, dn_ = shorts)."""
    rows = []
    for k, s in g.groupby(key):
        row = {"period": k}
        for pre, sign in [("up", 1), ("dn", -1)]:
            t = s[s["sign"] == sign]
            st = stats(t, name, 1.0)
            row.update({f"{pre}_ev": st["events"], f"{pre}_ret": st.get("ret", np.nan),
                        f"{pre}_hedged": st.get("hedged", np.nan), f"{pre}_t": st.get("t", np.nan)})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.add_argument("--start", type=str, default=None, help="First gap date")
    parser.add_argument("--end", type=str, default=None, help="Last gap date")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Only once the rule is frozen.")
    parser.add_argument("--min-gap", type=float, default=MIN_GAP, help="Gap at the open, %% from the prior close (5)")
    parser.add_argument("--min-vol-x", type=float, default=MIN_VOL_X,
                        help="Day 0 volume vs the 50 days before (3)")
    parser.add_argument("--dedupe", type=int, default=DEDUPE, help="A stock's first gap in this many bars (20)")
    parser.add_argument("--pick", type=str, default=PICK,
                        help="entry:hold, entry close (day 0 close) or open (day 1 open), hold in days (close:20)")
    parser.add_argument("--borrow-pct", type=float, default=5.0, help="Borrow fee for the shorts, %% a year")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check on a chart")
    parser.add_argument("--save-events", type=str, default=None, help="Write every event to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    entry, hold = parse_pick(args.pick)

    t0 = time.time()
    cols = (rules.universe_columns(P) + ["open", "high", "low", "close", "volume", "dv50", "sma200", "bad_bar",
                                         "bad_bar_dn", "in_sp500", "in_sp400", "in_sp600"])
    load_from = (pd.Timestamp(args.start) - pd.Timedelta(days=LOAD_PAD_DAYS)) if args.start else None
    try:
        df = datastore.load_panel(cols, start=load_from, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=120)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python pipeline/build_data.py, then python pipeline/build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy() if hasattr(df["symbol"], "cat") else \
        pd.factorize(df["symbol"])[0]
    market = pd.read_parquet(config.MARKET_FILE, columns=["date", "spx_close"])
    spx = market.sort_values("date").set_index("date")["spx_close"]
    spx200 = datastore.load_market(["spx_above200"]).set_index("date")["spx_above200"]
    keep_market = None
    if rules.market_columns(P):
        keep_market = rules.market_mask(df["date"], datastore.load_market(rules.market_columns(P)), P)
    ev, counts = select(df, codes, P, args, keep_market)
    first, _ = stock_bounds(codes)
    ev["usual_vol"] = usual_volume(df["volume"].to_numpy(), first)[ev["row"].to_numpy()]

    print("=" * 110)
    print(f"BIG-GAP FOLLOW-THROUGH  gap {args.min_gap:g}%+ at the open on {args.min_vol_x:g}x+ volume  "
          f"universe={args.universe}  start={args.start}  end={args.end}  market={P['market']}  "
          f"liq top {P['liq_top_pct']:g}%  min price ${P['min_price']:g}  size={P['size']}  "
          f"borrow {args.borrow_pct:g}%/yr  ({time.time() - t0:.1f}s)")
    print("=" * 110)
    print("Gaps found: " + ", ".join(f"{v:,} {k}" for k, v in counts.items())
          + f". Passed the universe at the prior close: {len(ev):,}.")
    if ev.empty:
        print("No events.")
        return
    years = ce.span_years(ev["date"])
    n_up, n_dn = int((ev["sign"] > 0).sum()), int((ev["sign"] < 0).sum())
    print(f"{len(ev):,} events, {years:.1f} years ({ev['date'].min():%Y-%m-%d} to {ev['date'].max():%Y-%m-%d}): "
          f"{n_up:,} up ({n_up / years:.0f} a year), {n_dn:,} down ({n_dn / years:.0f} a year). "
          f"Median gap {ev['gap'].abs().median():.1f}%, median volume {ev['vol_x'].median():.1f}x usual.")
    pd.set_option("display.width", 250)

    ev = features(ev, df, spx200)
    for label, sign in SIDES:
        g = ev[ev["sign"] == sign]
        print(f"\nDay by day around the {label} gaps ({len(g):,}), in the gap's direction: vs_spx = average move "
              f"against the S&P that day, % (+ = with the gap); running = their total from the day 0 close, the "
              f"drift a trade in at the close would catch; with_% = share moving with the gap; med_vol_x = median "
              f"volume vs usual. Day 0: a {g['gap'].abs().mean():.2f}% gap on average, then "
              f"{g['after_open'].mean():+.2f}% from the open to the close in the gap's direction.")
        print(profile(ev, df, spx, sign).round(2).to_string(index=False))

    grid = [(e, hh) for e in ENTRIES for hh in HOLDS]
    if (entry, hold) not in grid:
        grid.append((entry, hold))
    cols_out = {}
    for e, hh in grid:
        cols_out.update(trades(ev, df, spx, e, hh, args.borrow_pct))
    ev = pd.concat([ev, pd.DataFrame(cols_out)], axis=1)
    cost = ce.round_trip_cost()
    print(f"\nEvery entry and hold, each direction (ret = after a {cost:.2f}% round trip and, for the shorts, "
          f"{args.borrow_pct:g}%/yr borrow; hedged = against the S&P; adv20_% = went {ADVERSE:g}%+ against the "
          f"trade before the exit):")
    rows = []
    for label, sign in SIDES:
        g = ev[ev["sign"] == sign]
        for e, hh in grid:
            rows.append({"gap": label, "trade": "buy" if sign > 0 else "short", "entry": e, "hold": hh,
                         **stats(g, f"{e}:{hh}", years)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    name = f"{entry}:{hold}"
    pick = ev[ev[f"ret_{name}"].notna()].reset_index(drop=True)
    print(f"\nPICK: in at the {'day 0 close' if entry == 'close' else 'day 1 open'}, out at the close of day {hold} "
          f"({len(pick):,} trades). up_ = buying the up gaps, dn_ = shorting the down gaps; ret / hedged as above, "
          f"t_h = month-clustered t of hedged. Splits use what was known at the day 0 close.")
    print(split_table(pick, name, years, args.min_gap, args.min_vol_x).round(2).to_string(index=False))

    print("\nPick by 5-year block (t = of ret):")
    print(by_period(pick, name, ce.block_of(pick["date"])).round(2).to_string(index=False))
    yr = by_period(pick, name, pick["date"].dt.year)
    print("\nPick by year:")
    print(yr.round(2).to_string(index=False))
    for label, pre in [("Buying up gaps", "up"), ("Shorting down gaps", "dn")]:
        y = yr[f"{pre}_ret"].dropna()
        print(f"{label}: positive in {int((y > 0).sum())} of {len(y)} years.")

    if args.sample:
        s = pick.sample(min(args.sample, len(pick)), random_state=0).sort_values("date")
        dates = df["date"].to_numpy()
        s = s.assign(exit_day=dates[np.minimum(s["row"] + hold, s["last"])])
        print(f"\n{len(s)} random pick trades to check on a chart (date = the gap day):")
        s = s[["symbol", "date", "exit_day", "gap", "vol_x", "price", "after_open", f"ret_{name}"]]
        print(s.round({"gap": 2, "vol_x": 2, "price": 2, "after_open": 2, f"ret_{name}": 2}).to_string(index=False))

    if args.save_events:
        keep = ["symbol", "date", "sign", "gap", "vol_x", "price", "dv_m", "index", "after_open", "day0", "run60",
                "above200", "spx_above200"] + [f"ret_{e}:{hh}" for e, hh in grid]
        ev[keep].to_csv(args.save_events, index=False)
        print(f"\nSaved {len(ev)} events to {args.save_events}")

    print(f"\nWorth building on, set before the first run: the pick's ret clearly above 0 after costs in one "
          f"direction, month-clustered t above 2, positive in more than half the years and in the recent blocks, "
          f"and big enough to matter (about 5% a trade, or a smaller edge with about 50+ trades a year). Grid rows "
          f"other than the pick are {len(grid) * 2 - 2} more tries, so read them as context.")

    if not args.no_log:
        res = {}
        for label, sign in SIDES:
            st = stats(pick[pick["sign"] == sign], name, years)
            res[label] = {k: (round(float(v), 3) if v is not None and np.isfinite(v) else None) for k, v in st.items()}
        research_log.append(
            question="Do liquid stocks keep going after a big gap on heavy volume?",
            phase="gap", split=args.universe, script="gap_events.py",
            params={"start": args.start, "end": args.end, "min_gap": args.min_gap, "min_vol_x": args.min_vol_x,
                    "dedupe": args.dedupe, "pick": args.pick, "market": P["market"], "size": P["size"],
                    "liq_top_pct": P["liq_top_pct"], "min_price": P["min_price"], "borrow_pct": args.borrow_pct},
            combinations=len(grid) * 2, runs=1, results=res)


if __name__ == "__main__":
    main()
