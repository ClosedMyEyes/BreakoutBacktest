"""
Reverse-split short — do stocks keep falling after a reverse split, by enough to pay the borrow?
=================================================================================================
Idea #2 in ideas/next_setups.md. Most reverse splits are forced: a Nasdaq stock that
closes under $1 for 30 business days gets a deficiency notice and about 180 days to
fix it, and a reverse split is the usual fix. Most of these companies burn cash and
sell new shares into any strength once the price is back up (offerings, at-the-market
programs, warrants). The spike and crash shorts both did better on stocks with a
reverse split in the past year, on small samples found after the fact. This checks it
as its own event, charging a high borrow fee from the day of entry, since these are
among the hardest stocks to borrow.

No setups or sizing and no stop on the pick: an event table, like the gap and lock-up
studies (--explore also tries stops).

  event     day 0 = the first bar at the new price: a listed bar where Norgate's factor
            (printed close / adjusted close) rose --min-ratio (1.4) times or more since
            the bar before, in the same stock. The ratio is the size of the jump (10 for a
            1-for-10). The stock needs 60 bars before day 0, and only its first reverse
            split in --dedupe (60) bars counts. A jump that falls back within 5 bars is a
            bad print, not a split, and is left out
  universe  at the day 0 close, with the printed price: $1+ (--min-price), top 80% by
            50-day dollar volume (--liq-top-pct), not pinned. No price break in the 252
            bars before day 0, judged at day -1, so a break on day 0 itself (a split day
            that also moved 25%+) is allowed and flagged
  trade     short with no stop, in at the day 0 close (close0), the day 1 open (open1),
            the day 5 close (close5) or the day 20 close (close20), and cover at the close
            H = 5 / 20 / 60 / 120 / 250 trading days after the entry close (open1: after
            the day 0 close, so it has close0's cover and differs only in the entry
            price). A stock that stops trading is covered at its last exchange close. A
            window across a downward price break is left out (an untrusted gain for a
            short); an upward one counts as a real loss. A trade still open when the data
            ends is left out
  pick      set before the first run: close0:60 judged at 50%/yr borrow (--pick close0:60
            --pick-fee 50). Every other row is context. After exploring Dev, the rule for the
            holdout was frozen on 2026-10-15 (--frozen): 1-for-20+ splits, not an index
            member, close0:20

Costs: a round trip (ce.round_trip_cost(), about 0.50%) plus the borrow fee for every
trading day held (days / 252), at each of --borrow-pcts (5, 25, 50, 100, 200 % a year).

Columns (in % of the entry price, a short earns minus the stock's move):
  ret        the short after a round trip and the borrow fee
  hedged     the same against the S&P (short the stock, buy the S&P); hedged_comp against
             the Nasdaq Composite. Neither is a real small-cap hedge
  med / win_% / worst5% / worst   median, share of winners, the 1-in-20 worst, the worst
  adv50_%    share that went 50%+ against the short (a daily high 50% above the entry)
             before the cover
  t / t_hedged   month-clustered t
  be_fee     break-even borrow fee, % a year: the row's average short before borrow over
             its average fraction of a year held. The fee the edge could pay and still
             break even
  delist_%   share that stopped trading during the hold (covered at the last close)

HOW TO RUN:
  python reverse_split/reverse_split_events.py --start 2010-01-01
  python reverse_split/reverse_split_events.py --start 2010-01-01 --sample 10
  python reverse_split/reverse_split_events.py --start 2010-01-01 --min-price 5        # the usual $5 floor
  python reverse_split/reverse_split_events.py --start 2010-01-01 --only ratio:5:20    # one bucket, by block
  python reverse_split/reverse_split_events.py --start 2010-01-01 --explore            # refine the selection on Dev
  python reverse_split/reverse_split_events.py --start 2010-01-01 --explore --only spx_above200:1:2   # within a filter
  python reverse_split/reverse_split_events.py --start 2010-01-01 --frozen --universe holdout --unlock-holdout  # once

The run prints, in order: the counts and data checks, the day-by-day move around day 0,
every entry and hold (with the fee ladder), the pick split by what was known at the
day 0 close, the pick by 5-year block and year, the losses and an account test, the
sample, and the pass bar with the number of tries.

--explore prints only the tables for refining the selection on Dev, each split by 5-year
block: the pick by downtrend, market, volume, split and size filters, every entry and hold,
and stops at +300 / +200 / +100 / +50% above the entry with an account test at 2% a short.
--only first narrows the events, so the tables show one filter inside another.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "crash_rebound"))
sys.path.insert(0, os.path.join(ROOT, "lockup_short"))

import config  # noqa: E402
import datastore  # noqa: E402
import research_log  # noqa: E402
import rules  # noqa: E402
from params import add_param_flags, collect  # noqa: E402
import crash_events as ce  # noqa: E402
import lockup_events as le  # noqa: E402

MIN_RATIO = 1.4                     # factor jump that counts as a reverse split (config.BREAK_FACTOR_JUMP)
MIN_BARS = 60                       # bars of history needed before day 0
DEDUPE = 60                         # a stock's first reverse split in this many bars
REVERT_BARS = 5                     # a jump undone within this many bars is a bad print
PRIOR_BARS = 756                    # "another reverse split in the past 3 years"
ENTRIES = {"close0": (0, "close"), "open1": (1, "open"), "close5": (5, "close"), "close20": (20, "close")}
HOLDS = [5, 20, 60, 120, 250]       # cover at the close this many trading days after the entry close
PICK = "close0:60"
PICK_FEE = 50.0                     # % a year: the fee the pick is judged at
BORROW_PCTS = "5,25,50,100,200"
STRESS_COST = 1.5                   # % round trip: wider spreads on $1-5 stocks
ADVERSE = 50.0                      # % against the short counted in adv50_%
PROFILE_DAYS = range(-60, 121)
PROFILE_SHOW = [-60, -50, -40, -30, -20, -15] + list(range(-10, 11)) + [15, 20, 30, 40, 50, 60, 80, 100, 120]
VOL_BASE = (-60, -11)               # the stock's usual volume, days before day 0
RUN_BARS = 60                       # the move before day 0
LOAD_PAD_DAYS = 1200                # calendar days loaded before --start: 3 years for the prior-split flag
TAIL_DAYS = 430                     # calendar days loaded after --end: close20 + 250 bars
RATIO_BINS = [5, 20]                # 1-for-5, 1-for-20
EXPLORE_RATIO_BINS = [5, 20, 50, 100]           # --explore splits the big ratios further (tiny floats)
SPIKE = 200.0                                   # x3_% = share whose high reached 3x the entry (+200%)
RATIO_DIST = [1.4, 2, 5, 10, 20, 50, 100, np.inf]
PRE_PRICE_EDGES = [0, 0.5, 1, np.inf]           # printed close on day -1, $
PRICE_BINS = [2, 5, 10]                         # printed close on day 0, $, above --min-price
DV_EDGES = [0, 1, 5, 20, np.inf]                # 50-day $ volume, $M a day
RUN_EDGES = [-np.inf, -50, -20, 0, np.inf]      # % move over the 60 bars before day 0
DAY0_BAND = 5.0                                 # day 0 move within this % of day -1 = flat
COVID = ("2020-03-01", "2021-12-31")            # the episode the pick must survive without
BLOCKS = ["2010-2014", "2015-2019", "2020-2024", "2025-2029"]   # the explore tables' block columns
STOPS = [300, 200, 100, 50]                     # --explore: % above the entry (no stop is the first row)
BIG_HOLD = 10 ** 6                              # lets lockup_events.account() take any cover row
NEED_BLOCKS = ["2015-2019", "2020-2024", "2025-2029"]
FROZEN_SET = "2026-10-15"                       # --frozen: the rule set for the one holdout run, after --explore
FROZEN_PICK = "close0:20"
FROZEN_ONLY = ["ratio:20:inf", "in_index:0:1"]  # 1-for-20 and up, not in the S&P 500/400/600 or Russell 2000
FILTERS = [                                     # the filter table: (label, keep), each known at the day 0 close
    ("not after the S&P rose 5%+ in 20 days", lambda g: g["spx_move20"] < 5),
    ("S&P above its 200-day", lambda g: g["spx_above200"] == 1),
    ("$1M+ a day traded", lambda g: g["dv_m"] >= 1),
    ("volume last 50 days 1.5x+ the 200 before", lambda g: g["vol50_x"] >= 1.5),
    ("90%+ below the 52-week high", lambda g: g["from_high"] >= 90),
    ("day 0 volume 2x+ usual", lambda g: g["vol0_x"] >= 2),
    ("down 90%+ over 250 days", lambda g: g["move250"] < -90),
    ("down 75%+ over 120 days", lambda g: g["move120"] < -75),
    ("an earlier reverse split in 3 years", lambda g: g["prior_n"] >= 1),
]
SIZE_UP = 2.0                                   # --frozen sizing check: earlier-split trades get this many x the size
SIZES = "2,5"                                   # account test: each short is X% of the account
BIG_RATIO = 100.0                               # account test: 1-for-100+ shorts are also shown at
BIG_SCALE = 0.5                                 # this share of the size (set after the holdout run)
WORST = 10


def parse_pick(spec):
    """'close0:60' -> ('close0', 60)."""
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


def factor_jumps(df, codes):
    """Norgate's factor (printed / adjusted close) and its move since the bar before in
    the same stock (NaN on a stock's first row). A reverse split moves it up by the ratio;
    a forward split, spin-off or special distribution moves it down."""
    u = df["unadj_close"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(c > 0, u / c, np.nan)
        same = np.r_[False, codes[1:] == codes[:-1]]
        jump = np.where(same, f / np.r_[np.nan, f[:-1]], np.nan)
    return f, jump


def find_splits(df, codes, min_ratio=MIN_RATIO, dedupe=DEDUPE, min_bars=MIN_BARS):
    """Reverse splits: listed rows where the factor jumped min_ratio times or more. Returns
    a frame with row / first / last / ratio / prior_split (another reverse split in the
    PRIOR_BARS bars before) for each stock's first split in `dedupe` bars with min_bars
    of history, and the counts of what was left out."""
    f, jump = factor_jumps(df, codes)
    first, last = stock_bounds(codes)
    listed = df["listed"].to_numpy() == 1
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero((jump >= min_ratio) & listed)
    # A bad print: the factor comes back to within min_ratio of where it was before
    back = np.zeros(len(cand), dtype=bool)
    before = f[np.maximum(cand - 1, 0)]
    for k in range(1, REVERT_BARS + 1):
        jj = cand + k
        ok = jj <= last[cand]
        with np.errstate(invalid="ignore"):
            back |= ok & (f[np.where(ok, jj, cand)] < before * min_ratio)
    counts = {"upward factor jumps on listed bars": int(len(cand)),
              f"bad prints (undone within {REVERT_BARS} bars)": int(back.sum())}
    cand = cand[~back]
    prev = np.r_[-10 ** 9, cand[:-1]]
    same = prev >= first[cand]
    repeat = same & (cand - prev <= dedupe)
    prior = same & (cand - prev <= PRIOR_BARS)
    lo = np.maximum(cand - PRIOR_BARS, first[cand])
    n_prior = np.arange(len(cand)) - np.searchsorted(cand, lo)      # earlier splits of the stock in the window
    short = (cand - first[cand] < min_bars) & ~repeat
    counts["repeats within the dedupe bars"] = int(repeat.sum())
    counts[f"under {min_bars} bars of history"] = int(short.sum())
    keep = ~repeat & ~short
    rows = cand[keep]
    ev = pd.DataFrame({"row": rows, "first": first[rows], "last": last[rows], "ratio": jump[rows],
                       "prior_split": prior[keep], "prior_n": n_prior[keep]})
    return ev, counts


def select(df, codes, P, args, keep_market=None):
    """Splits inside the dates whose stock passed the universe: price and liquidity at the
    day 0 close, the price-break rule at day -1 (so a break on day 0 is allowed). Returns
    (events, counts)."""
    ev, counts = find_splits(df, codes, args.min_ratio, args.dedupe)
    d = df["date"].to_numpy()
    ev["date"] = d[ev["row"].to_numpy()]
    if args.start:
        ev = ev[ev["date"] >= pd.Timestamp(args.start)]
    if args.end:
        ev = ev[ev["date"] <= pd.Timestamp(args.end)]
    ev = ev.reset_index(drop=True)
    counts["inside the dates"] = len(ev)
    r = ev["row"].to_numpy()
    sub = df.iloc[np.unique(r)] if len(r) else df.iloc[:0]
    day0 = pd.Series(rules.universe_mask(sub, {**P, "block_after_break": 0}), index=sub.index)
    ok = day0.reindex(r).fillna(False).to_numpy(dtype=bool).copy()
    block = int(P["block_after_break"])
    brk_ok = np.ones(len(r), dtype=bool)
    if block > 0 and len(r):
        bsb = df["bars_since_break"].to_numpy(np.float64)[r - 1]
        with np.errstate(invalid="ignore"):
            brk_ok = ~(bsb < block)                     # NaN = never broke
    px0 = df["unadj_close"].to_numpy(np.float64)[r]
    dvp = df["dv_pct"].to_numpy(np.float64)[r]
    with np.errstate(invalid="ignore"):
        counts["failed: under the price floor at day 0"] = int((~(px0 >= P["min_price"])).sum())
        counts["failed: below the liquidity cut at day 0"] = int((~(dvp >= 100 - P["liq_top_pct"])).sum())
    counts[f"failed: price break in the {block} bars before day 0"] = int((~brk_ok).sum())
    ok &= brk_ok
    if keep_market is not None:
        ok &= keep_market[r]
    ev = ev[ok].reset_index(drop=True)
    ev["symbol"] = df["symbol"].astype(str).to_numpy()[ev["row"].to_numpy()]
    # Stopped trading before the data ends (delisted or moved to OTC), not just still trading
    end = pd.Timestamp(d.max()) - pd.Timedelta(days=config.DELISTED_GRACE_DAYS)
    ev["gone"] = pd.DatetimeIndex(d[ev["last"].to_numpy()]) < end
    return ev, counts


def window_mean(x, r, a, b, first):
    """Mean of x over rows r+a .. r+b (same stock), NaN if r+a is before the stock's first row."""
    fin = np.isfinite(x)
    cs = np.r_[0.0, np.cumsum(np.where(fin, x, 0.0))]
    cn = np.r_[0, np.cumsum(fin)]
    lo, hi = r + a, r + b + 1
    ok = lo >= first
    lo, hi = np.where(ok, lo, r), np.where(ok, hi, r + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (n > 0), (cs[hi] - cs[lo]) / n, np.nan)


def features(ev, df, spx_above200, market=None, mkt=None):
    """What is known at the day 0 close, for the split table. With market (market features by
    date) and mkt (index closes by date), also the explore columns: the longer downtrend, the
    volume before day 0 and the market at day 0."""
    c = df["close"].to_numpy(np.float64)
    u = df["unadj_close"].to_numpy(np.float64)
    r, first = ev["row"].to_numpy(), ev["first"].to_numpy()
    p = r - 1
    ev["px_prev"], ev["px0"] = u[p], u[r]
    ev["dv_m"] = df["dv50"].to_numpy(np.float64)[r] / 1e6
    sp = df[["in_sp500", "in_sp400", "in_sp600"]].to_numpy()[r].max(axis=1) > 0
    r2 = df["in_r2000"].to_numpy()[r] > 0
    ev["index"] = np.select([sp, r2], ["S&P 500/400/600", "Russell 2000 only"], "neither")
    ev["in_index"] = (sp | r2).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = r - RUN_BARS                            # the first of the 60 bars before day 0
        ev["run60"] = np.where(a >= first, (c[p] / c[np.maximum(a, 0)] - 1) * 100, np.nan)
        ev["day0"] = (c[r] / c[p] - 1) * 100
    ev["break0"] = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()[r]
    sma = df["sma200"].to_numpy(np.float64)[r]
    ev["above200"] = np.where(np.isfinite(sma), (c[r] > sma).astype(float), np.nan)
    ev["spx_above200"] = spx_above200.reindex(pd.DatetimeIndex(df["date"].to_numpy()[r])).to_numpy()
    if market is None:
        return ev
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in [120, 250]:
            a = r - k
            ev[f"move{k}"] = np.where(a >= first, (c[p] / c[np.maximum(a, 0)] - 1) * 100, np.nan)
        hi = df["hi52_c"].to_numpy(np.float64)[p]
        ev["from_high"] = np.where(hi > 0, (1 - c[p] / hi) * 100, np.nan)
        s50 = df["sma50"].to_numpy(np.float64)[r]
        ev["above50"] = np.where(np.isfinite(s50), (c[r] > s50).astype(float), np.nan)
        sh = df["turnover"].to_numpy(np.float64) / np.where(c > 0, c, np.nan)     # split-consistent shares
        usual = window_mean(sh, r, VOL_BASE[0], VOL_BASE[1], first)
        ev["vol10_x"] = window_mean(sh, r, -10, -1, first) / usual
        ev["vol50_x"] = window_mean(sh, r, -50, -1, first) / window_mean(sh, r, -250, -51, first)
        ev["vol0_x"] = sh[r] / usual
    d = pd.DatetimeIndex(df["date"].to_numpy()[r])
    m = market.reindex(d)
    for k in ["spx_above50", "comp_above200"]:
        ev[k] = m[k].to_numpy(np.float64)
    up = (m["spx_above200"] == 1) & (m["spx_above50"] == 1) & (m["spx_sma50_rising"] == 1)
    ev["spx_trend"] = up.to_numpy(float)
    ev["breadth"] = m["pct_above200"].to_numpy(np.float64)
    sc = mkt["spx_close"]
    ev["spx_move20"] = ((sc / sc.shift(20) - 1) * 100).reindex(d).to_numpy()
    return ev


def trades(ev, df, mkt, entry, hold, stop_pct=0.0):
    """The short for one entry and hold, before any borrow. Returns columns named entry:hold:
    g_ (short after the round trip, %), h_ / n_ (the same hedged with the S&P / Nasdaq
    Composite), y_ (years held, days / 252), adv_ (went ADVERSE% against), del_ (stopped
    trading during the hold), raw_ (the stock's move, %), peak_ (highest high above the
    entry, %) and exit_ (the cover row). The short at fee F is g_ - F x y_.
    stop_pct > 0: cover on the first day whose high reaches stop_pct% above the entry, at the
    stop (or the open, when it opened above it); columns are then named entry:hold/sN and add
    stop_ (1 if stopped)."""
    off, kind = ENTRIES[entry]
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    cdn = np.cumsum((df["bad_bar_dn"].fillna(0) > 0).to_numpy())
    dates = df["date"].to_numpy()
    r, last, gone = ev["row"].to_numpy(), ev["last"].to_numpy(), ev["gone"].to_numpy(dtype=bool)
    e = r + off                                     # the entry bar
    base = r if kind == "open" else e               # the close the hold counts from
    target = base + hold
    j = np.minimum(target, last)
    ok = (e <= last) & (j > base)
    ee = np.minimum(e, last)
    px = np.where(ok, o[ee] if kind == "open" else c[ee], np.nan)
    ok &= px > 0
    cut = target > last
    level = px * (1 + stop_pct / 100) if stop_pct > 0 else np.full(len(r), np.inf)
    out_row, out_px = j.copy(), c[j].copy()
    hit_any = np.zeros(len(r), dtype=bool)
    peak = np.full(len(r), np.nan)
    for k in range(0 if kind == "open" else 1, int((target - e).max(initial=0)) + 1):
        jj = np.minimum(e + k, last)
        live = (e + k <= j) & ~hit_any
        peak = np.where(live, np.fmax(peak, h[jj]), peak)
        if stop_pct > 0:
            hit = live & (h[jj] >= level)
            fill = level if (kind == "open" and k == 0) else np.fmax(o[jj], level)
            out_row, out_px = np.where(hit, jj, out_row), np.where(hit, fill, out_px)
            hit_any |= hit
    peak = (peak / px - 1) * 100
    j = out_row
    bad = ~ok | (cut & ~gone & ~hit_any) | (cdn[j] - cdn[base] > 0)
    raw = out_px / px - 1
    pos = mkt.index.get_indexer(pd.DatetimeIndex(dates))
    pe, pj = pos[ee], pos[j]
    have = (pe >= 0) & (pj >= 0)
    idx = {}
    for k in ["spx", "comp"]:
        m_in = mkt[f"{k}_{kind}"].to_numpy(np.float64)[np.maximum(pe, 0)]
        m_out = mkt[f"{k}_close"].to_numpy(np.float64)[np.maximum(pj, 0)]
        idx[k] = np.where(have, m_out / m_in - 1, np.nan)
    days = np.maximum(j - base, 1)
    cost = ce.round_trip_cost()
    name = f"{entry}:{hold}" + (f"/s{stop_pct:g}" if stop_pct > 0 else "")
    nan_if_bad = lambda x: np.where(bad, np.nan, x)
    out = {f"stop_{name}": nan_if_bad(hit_any.astype(float))} if stop_pct > 0 else {}
    return {**out, f"g_{name}": nan_if_bad(-raw * 100 - cost),
            f"h_{name}": nan_if_bad(-(raw - idx["spx"]) * 100 - cost),
            f"n_{name}": nan_if_bad(-(raw - idx["comp"]) * 100 - cost),
            f"y_{name}": nan_if_bad(days / 252),
            f"adv_{name}": nan_if_bad((peak >= ADVERSE).astype(float)),
            f"del_{name}": nan_if_bad((cut & gone & ~hit_any).astype(float)),
            f"raw_{name}": nan_if_bad(raw * 100),
            f"peak_{name}": nan_if_bad(peak),
            f"exit_{name}": j}


def ret(g, name, fee, col="g"):
    """The short (col g), hedged (h) or hedged with the Nasdaq (n) at fee % a year."""
    return g[f"{col}_{name}"] - fee * g[f"y_{name}"]


def breakeven(g, name):
    """The borrow fee (% a year) at which the row's average short is 0."""
    s = g[[f"g_{name}", f"y_{name}"]].dropna()
    y = s[f"y_{name}"].mean() if len(s) else np.nan
    return s[f"g_{name}"].mean() / y if len(s) and y > 0 else np.nan


def stats(g, name, fee, years):
    s = ret(g, name, fee).dropna() if len(g) else pd.Series(dtype=float)
    out = {"trades": len(s), "per_yr": len(s) / years}
    if s.empty:
        return out
    out["ret"] = s.mean()
    out["hedged"] = ret(g, name, fee, "h").mean()
    out["hedged_comp"] = ret(g, name, fee, "n").mean()
    out["med"] = s.median()
    out["win_%"] = (s > 0).mean() * 100
    out["worst5%"] = s.quantile(0.05)
    out["worst"] = s.min()
    out["adv50_%"] = g[f"adv_{name}"].mean() * 100
    out["t"] = ce.tstat_by_month(ret(g, name, fee), g["date"])
    out["t_hedged"] = ce.tstat_by_month(ret(g, name, fee, "h"), g["date"])
    out["be_fee"] = breakeven(g, name)
    out["delist_%"] = g[f"del_{name}"].mean() * 100
    return out


def profile(ev, df, mkt):
    """Day by day around day 0 against the S&P, in the short's favour: the average move
    (vs_spx, + = the stock fell more than the S&P), the share moving the short's way, the
    median volume against usual (turnover / adjusted close, so split-consistent), the
    share back under $1 printed and the share that had stopped trading by that day."""
    c = df["close"].to_numpy(np.float64)
    u = df["unadj_close"].to_numpy(np.float64)
    tv = df["turnover"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(c > 0, tv / c, np.nan)
    brk = ((df["bad_bar"].fillna(0) > 0) | (df["bad_bar_dn"].fillna(0) > 0)).to_numpy()
    pos = mkt.index.get_indexer(pd.DatetimeIndex(df["date"].to_numpy()))
    sc = mkt["spx_close"].to_numpy(np.float64)
    r, first, last, gone = (ev[k].to_numpy() for k in ["row", "first", "last", "gone"])
    gone = gone.astype(bool)
    usual = np.full(len(r), np.nan)
    for k in range(len(r)):
        a, b = r[k] + VOL_BASE[0], r[k] + VOL_BASE[1]
        if a >= first[k]:
            x = sh[a:b + 1]
            usual[k] = np.nanmean(x) if np.isfinite(x).any() else np.nan
    rows = []
    for d in PROFILE_DAYS:
        j = r + d
        on = (j - 1 >= first) & (j <= last)
        jj, ii = np.where(on, j, r), np.where(on, j - 1, r)
        pi, pj = pos[ii], pos[jj]
        sp = np.where((pi >= 0) & (pj >= 0), sc[np.maximum(pj, 0)] / sc[np.maximum(pi, 0)] - 1, np.nan)
        x = np.where(on & ~brk[jj], -(c[jj] / c[ii] - 1 - sp) * 100, np.nan)
        fin = np.isfinite(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = np.where(on, sh[jj] / usual, np.nan)
        base = (j <= last) | (gone & (j > last))       # known: still trading, or already gone
        live = (j <= last) & (j >= first)
        rows.append({"day": d, "vs_spx": x[fin].mean() if fin.any() else np.nan,
                     "short_way_%": (x[fin] > 0).mean() * 100 if fin.any() else np.nan,
                     "med_vol_x": np.nanmedian(vx) if np.isfinite(vx).any() else np.nan,
                     "under1_%": (u[jj][live] < 1).mean() * 100 if live.any() else np.nan,
                     "delisted_%": (j[base] > last[base]).mean() * 100 if base.any() else np.nan})
    out = pd.DataFrame(rows)
    v = out["vs_spx"].fillna(0).to_numpy()
    days = out["day"].to_numpy()
    after = np.cumsum(np.where(days > 0, v, 0))
    before = np.cumsum(np.where(days <= 0, v, 0)[::-1])[::-1]
    out.insert(2, "running", np.where(days > 0, after, before))
    return out


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


def keep_only(ev, specs):
    """Keep the events whose column is in [lo, hi) for every 'col:lo:hi' spec (-inf / inf
    allowed), e.g. ratio:5:20."""
    for spec in specs:
        col, lo, hi = spec.split(":")
        if col not in ev.columns:
            raise SystemExit(f"--only: no column {col}")
        v = ev[col].astype(float)
        ev = ev[(v >= float(lo)) & (v < float(hi))]
    return ev.reset_index(drop=True)


def split_groups(g, min_ratio, min_price):
    """The buckets of the split table: what was known at the day 0 close."""
    money = lambda x: f"${x:g}"
    pct = lambda x: f"{x:g}%"
    one_for = lambda x: f"1-for-{x:g}"
    day0 = g["day0"]
    yes_no = lambda m: [("yes", m), ("no", ~m)]
    return [
        ("all", [("", np.ones(len(g), dtype=bool))]),
        ("ratio", bins_of(g["ratio"], edges_of(min_ratio, RATIO_BINS), one_for)),
        ("printed price, day -1", bins_of(g["px_prev"], PRE_PRICE_EDGES, money)),
        ("printed price, day 0", bins_of(g["px0"], edges_of(min_price, PRICE_BINS), money)),
        ("$ volume, $M/day (50-day)", bins_of(g["dv_m"], DV_EDGES, money)),
        ("index", [(k, (g["index"] == k).to_numpy()) for k in ["S&P 500/400/600", "Russell 2000 only", "neither"]]),
        (f"{RUN_BARS}-day move before day 0", bins_of(g["run60"], RUN_EDGES, pct)),
        ("day 0 move from the day -1 close", [(f"down {DAY0_BAND:g}%+", (day0 <= -DAY0_BAND).to_numpy()),
                                              ("flat", (day0.abs() < DAY0_BAND).to_numpy()),
                                              (f"up {DAY0_BAND:g}%+", (day0 >= DAY0_BAND).to_numpy())]),
        ("another reverse split in the past 3 years", yes_no(g["prior_split"].to_numpy(dtype=bool))),
        ("day 0 is a price break", yes_no(g["break0"].to_numpy(dtype=bool))),
        ("stock vs its 200-day", [("above", (g["above200"] == 1).to_numpy()),
                                  ("below", (g["above200"] == 0).to_numpy())]),
        ("S&P vs its 200-day", [("above", (g["spx_above200"] == 1).to_numpy()),
                                ("below", (g["spx_above200"] == 0).to_numpy())]),
    ]


def split_table(g, name, fee, low_fee, years, groups):
    rows = []
    for group, buckets in groups:
        for label, mask in buckets:
            s = g.loc[np.asarray(mask, dtype=bool)]
            st = stats(s, name, fee, years)
            rows.append({"group": group, "bucket": label, "trades": st["trades"], "per_yr": st["per_yr"],
                         f"ret@{low_fee:g}": ret(s, name, low_fee).mean() if len(s) else np.nan,
                         **{k: st.get(k, np.nan) for k in ["ret", "hedged", "hedged_comp", "win_%", "adv50_%",
                                                           "t", "be_fee"]}})
    return pd.DataFrame(rows)


def by_period(g, name, fee, key):
    rows = []
    for k, s in g.groupby(key):
        st = stats(s, name, fee, 1.0)
        rows.append({"period": k, **{c: st.get(c, np.nan) for c in ["trades", "ret", "hedged", "hedged_comp",
                                                                     "win_%", "t", "be_fee"]}})
    return pd.DataFrame(rows)


def explore_groups(g, min_ratio):
    """The --explore families: (family, [(filter, [(bucket, mask)])]), all known at the day 0 close."""
    pct = lambda x: f"{x:g}%"
    xs = lambda x: f"{x:g}x"
    money = lambda x: f"${x:g}"
    flag = lambda col, yes, no: [(yes, (g[col] == 1).to_numpy()), (no, (g[col] == 0).to_numpy())]
    pn = g["prior_n"]
    return [
        ("DOWNTREND (the stock before day 0)", [
            ("60-day move", bins_of(g["run60"], RUN_EDGES, pct)),
            ("120-day move", bins_of(g["move120"], [-np.inf, -75, -50, -25, 0, np.inf], pct)),
            ("250-day move", bins_of(g["move250"], [-np.inf, -90, -75, -50, 0, np.inf], pct)),
            ("% below 52-week high", bins_of(g["from_high"], [0, 50, 75, 90, np.inf], pct)),
            ("vs its 200-day", flag("above200", "above", "below")),
            ("vs its 50-day", flag("above50", "above", "below"))]),
        ("MARKET (at the day 0 close)", [
            ("S&P vs 200-day", flag("spx_above200", "above", "below")),
            ("S&P vs 50-day", flag("spx_above50", "above", "below")),
            ("S&P uptrend", flag("spx_trend", "yes", "no")),
            ("S&P 20-day move", bins_of(g["spx_move20"], [-np.inf, -5, 0, 5, np.inf], pct)),
            ("Nasdaq vs 200-day", flag("comp_above200", "above", "below")),
            ("% of stocks > 200d", bins_of(g["breadth"], [0, 40, 60, np.inf], pct))]),
        ("VOLUME (split-consistent shares)", [
            ("last 10 days vs usual", bins_of(g["vol10_x"], [0, 0.75, 1, 1.5, np.inf], xs)),
            ("last 50 vs 200 before", bins_of(g["vol50_x"], [0, 0.75, 1, 1.5, np.inf], xs)),
            ("day 0 vs usual", bins_of(g["vol0_x"], [0, 2, 5, np.inf], xs))]),
        ("THE SPLIT", [
            ("ratio", bins_of(g["ratio"], edges_of(min_ratio, EXPLORE_RATIO_BINS), lambda x: f"1-for-{x:g}")),
            ("splits in past 3 yrs", [("none", (pn == 0).to_numpy()), ("1", (pn == 1).to_numpy()),
                                      ("2+", (pn >= 2).to_numpy())]),
            ("printed price day -1", bins_of(g["px_prev"], PRE_PRICE_EDGES, money)),
            ("day 0 move", [(f"down {DAY0_BAND:g}%+", (g["day0"] <= -DAY0_BAND).to_numpy()),
                            ("flat", (g["day0"].abs() < DAY0_BAND).to_numpy()),
                            (f"up {DAY0_BAND:g}%+", (g["day0"] >= DAY0_BAND).to_numpy())])]),
        ("SIZE", [
            ("$ volume, $M/day", bins_of(g["dv_m"], DV_EDGES, money)),
            ("index", [(k, (g["index"] == k).to_numpy())
                       for k in ["S&P 500/400/600", "Russell 2000 only", "neither"]])]),
    ]


def block_row(s, name, fee):
    """The trade `name` at fee for one set of trades: all years and each 5-year block."""
    st = stats(s, name, fee, 1.0)
    row = {"trades": st["trades"], "ret": st.get("ret", np.nan), "t": num(st.get("t")),
           "hedged": st.get("hedged", np.nan)}
    up = n = 0
    blk = ret(s, name, fee).groupby(ce.block_of(s["date"])).agg(["mean", "count"]) if len(s) else None
    for b in BLOCKS:
        v = blk["mean"].get(b, np.nan) if blk is not None else np.nan
        row[b[:4] + ("+" if b == BLOCKS[-1] else "-" + b[7:9])] = v
        if blk is not None and blk["count"].get(b, 0) > 0:
            n += 1
            up += int(v > 0)
    row["blocks_up"] = f"{up}/{n}"
    for k in ["win_%", "adv50_%", "be_fee"]:
        row[k] = st.get(k, np.nan)
    pk = s[f"peak_{name}"].where(s[f"g_{name}"].notna()).dropna() if len(s) else pd.Series(dtype=float)
    row["x3_%"] = (pk >= SPIKE).mean() * 100 if len(pk) else np.nan
    row["worst"] = st.get("worst", np.nan)
    return row


def explore(pick, ev, df, mkt, name, fee, grid, args):
    """--explore: tables for refining the selection on Dev, each split by 5-year block. Prints
    them and returns the number of rows (tries)."""
    tried = 0
    print(f"\nEXPLORE ({len(pick):,} trades of {args.pick} at {fee:g}%/yr borrow). Each row: trades, ret and its "
          f"month-clustered t, hedged with the S&P, the average ret in each 5-year block, blocks_up = blocks with "
          f"ret above 0, win_%, adv50_% (went {ADVERSE:g}%+ against first), be_fee, x3_% (the high reached 3x the "
          f"entry before the cover) and the worst trade. Everything is known at the day 0 "
          f"close. Many rows will look good by chance: a filter is worth keeping only if it holds in every block "
          f"and there is a reason for it.")
    for family, filters in explore_groups(pick, args.min_ratio):
        rows = []
        for flt, buckets in filters:
            for label, mask in buckets:
                rows.append({"filter": flt, "bucket": label, **block_row(pick.loc[mask], name, fee)})
                tried += 1
        print(f"\n{family}")
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    rows = [{"entry": e, "hold": hh, **block_row(ev[ev[f"g_{e}:{hh}"].notna()], f"{e}:{hh}", fee)} for e, hh in grid]
    tried += len(grid)
    print(f"\nEVERY ENTRY AND HOLD by block, at {fee:g}%/yr: does a shorter or longer hold hold up better?")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    off, kind = ENTRIES[parse_pick(args.pick)[0]]
    hold = parse_pick(args.pick)[1]
    rows = []
    for stop in [0] + STOPS:
        if stop:
            t = pd.concat([pick, pd.DataFrame(trades(pick, df, mkt, parse_pick(args.pick)[0], hold, stop))], axis=1)
            sname = f"{name}/s{stop:g}"
            t = t[t[f"g_{sname}"].notna()]
        else:
            t, sname = pick, name
        row = {"stop": f"+{stop:g}%" if stop else "none", **block_row(t, sname, fee)}
        r_ = ret(t, sname, fee)
        row["stopped_%"] = t[f"stop_{sname}"].mean() * 100 if stop else 0.0
        row["worst"], row["worst5%"] = r_.min(), r_.quantile(0.05)
        if kind == "close" and len(t):
            exit_row = t[f"exit_{sname}"].to_numpy()
            acc = pd.DataFrame({"row": t["row"].to_numpy() + off, "last": exit_row, "date": t["date"].to_numpy(),
                                f"short_0:{BIG_HOLD}": r_.to_numpy()})
            curve, _ = le.account(acc, df, mkt["spx_close"], 0, BIG_HOLD, 2.0, fee)
            if not curve.empty:
                a = le.account_summary(curve, 2.0)[0]
                row["acct2_cagr"], row["acct2_maxdd"] = a["cagr_%"], a["max_dd_%"]
        rows.append(row)
        tried += 1 if stop else 0
    print("\nSTOPS on the pick: cover when the day's high reaches the stop above the entry, at the stop (or the open, "
          "when it gapped above it). The tail is the risk here (adv50_% above), so wide stops are tested too. "
          "acct2_ = the account test at 2% of the account a short: yearly growth and the worst drop.")
    out = pd.DataFrame(rows)
    out = out.drop(columns=[c for c in ["win_%", "adv50_%", "be_fee"] if c in out])
    print(out.round(2).to_string(index=False))
    return tried


def apply_frozen(args):
    """--frozen: the pick and filter set in the README on FROZEN_SET for the holdout run."""
    if args.frozen:
        if args.only or args.pick is not None:
            raise SystemExit("--frozen sets --pick and --only itself; leave them out.")
        args.pick, args.only = FROZEN_PICK, list(FROZEN_ONLY)
    args.pick = args.pick or PICK
    return args


def filter_table(g, name, fee):
    """Each FILTERS row on the trades in g: what keeping only the trades that pass it does to the count, the
    average and the total (trades x average, % of one position: the profit if every trade is the same size)."""
    r = ret(g, name, fee)
    ok = r.notna().to_numpy()
    total = r[ok].sum()
    rows = [{"filter": "all trades", "trades": int(ok.sum()), "kept_%": 100.0, "ret": r[ok].mean(),
             "t": num(ce.tstat_by_month(r[ok], g["date"][ok])), "blocks_up": block_row(g[ok], name, fee)["blocks_up"],
             "ret_dropped": np.nan, "total": total, "total_vs_all_%": 100.0}]
    for label, keep in FILTERS:
        k = keep(g).fillna(False).to_numpy(dtype=bool) & ok
        d = ~k & ok
        rows.append({"filter": label, "trades": int(k.sum()), "kept_%": k.sum() / max(ok.sum(), 1) * 100,
                     "ret": r[k].mean(), "t": num(ce.tstat_by_month(r[k], g["date"][k])) if k.any() else np.nan,
                     "blocks_up": block_row(g[k], name, fee)["blocks_up"], "ret_dropped": r[d].mean(),
                     "total": r[k].sum(), "total_vs_all_%": r[k].sum() / total * 100 if total else np.nan})
    return pd.DataFrame(rows)


def sizing_check(g, name, fee):
    """Trades with an earlier reverse split in 3 years against the rest: (n, ret) each, the difference and its
    Welch t (trades as independent)."""
    r = ret(g, name, fee)
    a, b = r[(g["prior_n"] >= 1) & r.notna()], r[(g["prior_n"] == 0) & r.notna()]
    if len(a) < 2 or len(b) < 2:
        return len(a), a.mean(), len(b), b.mean(), np.nan, np.nan
    diff = a.mean() - b.mean()
    return len(a), a.mean(), len(b), b.mean(), diff, diff / np.sqrt(a.var() / len(a) + b.var() / len(b))


def account_sized(g, df, spx, a, b, size_pct, borrow_pct, scale):
    """lockup_events.account() with each short opened at size_pct x its scale (one per row of g, so 0.5 is half
    the size). All 1 gives the same curve."""
    name = f"{a}:{b}"
    keep = g[f"short_{name}"].notna().to_numpy()
    g, scale = g[keep], np.asarray(scale, dtype=np.float64)[keep]
    c = df["close"].to_numpy(np.float64)
    dpos = spx.index.get_indexer(pd.DatetimeIndex(df["date"].to_numpy()))
    r, last, final = g["row"].to_numpy(), g["last"].to_numpy(), g[f"short_{name}"].to_numpy()
    i, j = r + a, np.minimum(r + b, last)
    on = (dpos[i] >= 0) & (dpos[j] >= 0)
    if not on.any():
        return pd.DataFrame(columns=["date", "equity", "open", "short_%"]), int((~on).sum())
    lo, hi = dpos[i][on].min(), dpos[j][on].max()
    n = hi - lo + 1
    pnl, n_open, value = np.zeros(n), np.zeros(n), np.zeros(n)
    cost = ce.round_trip_cost()
    for k in np.flatnonzero(on)[np.argsort(dpos[i][on], kind="stable")]:
        rows = np.arange(i[k], j[k] + 1)
        t = dpos[rows]
        rows, t = rows[t >= 0], t[t >= 0] - lo
        move = c[rows] / c[i[k]] - 1
        path = -move * 100 - cost / 2 - borrow_pct * (rows - i[k]) / 252
        path[-1] = final[k]
        size = size_pct * scale[k] / 100 * (1 + pnl[t[0]])
        span = np.full(t[-1] - t[0] + 1, np.nan)
        span[t - t[0]], held = path, np.full(t[-1] - t[0] + 1, np.nan)
        held[t - t[0]] = 1 + move
        span, held = pd.Series(span).ffill().to_numpy(), pd.Series(held).ffill().to_numpy()
        pnl[t[0]:t[-1] + 1] += size * span / 100
        pnl[t[-1] + 1:] += size * path[-1] / 100
        n_open[t[0]:t[-1]] += 1
        value[t[0]:t[-1]] += size * held[:-1]
    eq = 1 + pnl
    curve = pd.DataFrame({"date": spx.index[lo:hi + 1], "equity": eq, "open": n_open,
                          "short_%": value / eq * 100})
    return curve, int((~on).sum())


def num(x):
    return np.nan if x is None else x


def counts_table(v, edges, fmt):
    return pd.DataFrame([{"bucket": label, "events": int(m.sum())} for label, m in bins_of(v, edges, fmt)])


def data_checks(ev, df):
    """Lines that check the day 0 bars and the volume adjustment before trusting the table."""
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    tv = df["turnover"].to_numpy(np.float64)
    r, last = ev["row"].to_numpy(), ev["last"].to_numpy()
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        g0 = o[r] / c[r - 1]
        n1 = r + 1 <= last
        g1 = o[np.minimum(r + 1, last)][n1] / c[r][n1]
        odd = (h[r] < np.fmax(o[r], c[r]) * 0.999) | (lo[r] > np.fmin(o[r], c[r]) * 1.001)
        out.append(f"Day 0 bars (adjusted prices): the open is {np.nanmedian(g0):.3f}x the day -1 close (median); "
                   f"{np.mean((g0 < 0.5) | (g0 > 2)) * 100:.1f}% are under 0.5x or over 2x, "
                   f"{np.mean((g0 < 0.8) | (g0 > 1.25)) * 100:.1f}% outside 0.8-1.25x. The day 1 open is "
                   f"{np.nanmedian(g1):.3f}x the day 0 close; {np.mean((g1 < 0.5) | (g1 > 2)) * 100:.1f}% under 0.5x "
                   f"or over 2x. Day 0 high under the open or close, or low above them: {np.mean(odd) * 100:.1f}%. "
                   f"An open adjusted differently from the close would show as about 1/ratio or ratio x, in many "
                   f"events.")
        ratio = np.where(c > 0, vol / (tv / c), np.nan)
        pre = np.concatenate([ratio[r - k] for k in range(1, 6)])
        post = np.concatenate([ratio[np.minimum(r + k, last)][r + k <= last] for k in range(1, 6)])
        out.append(f"Volume: Norgate's volume over turnover / adjusted close is {np.nanmedian(pre):.2f} (median, days "
                   f"-5 to -1) and {np.nanmedian(post):.2f} (days 1 to 5). The same on both sides = volume is split-"
                   f"adjusted. The volume column below uses turnover / adjusted close either way, and dv50 is from "
                   f"turnover, so neither moves at the split.")
    return out


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.set_defaults(min_price=1.0, liq_top_pct=80.0)
    parser.add_argument("--start", type=str, default=None, help="First day 0 date")
    parser.add_argument("--end", type=str, default=None, help="Last day 0 date")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Only once the rule is frozen.")
    parser.add_argument("--pick", type=str, default=None,
                        help="entry:hold, entry close0 / open1 / close5 / close20, hold in days (close0:60)")
    parser.add_argument("--pick-fee", type=float, default=PICK_FEE,
                        help="Borrow fee the pick is judged at, %% a year (50)")
    parser.add_argument("--dedupe", type=int, default=DEDUPE,
                        help="A stock's first reverse split in this many bars (60)")
    parser.add_argument("--min-ratio", type=float, default=MIN_RATIO,
                        help="Factor jump that counts as a reverse split (1.4 = about 1-for-1.5 and up)")
    parser.add_argument("--borrow-pcts", type=str, default=BORROW_PCTS,
                        help="Borrow fees for the fee ladder, %% a year (5,25,50,100,200); the first is the main "
                             "table's")
    parser.add_argument("--only", action="append", default=[],
                        help="Keep events with a column in [lo, hi), e.g. --only ratio:5:20 (columns: ratio, px_prev, "
                             "px0, dv_m, run60, day0, prior_n, and for flags 1:2 = yes, e.g. spx_above200:1:2 or "
                             "in_index:0:1 for not an index member; also "
                             "move120, move250, from_high, above50, vol10_x, vol50_x, vol0_x, spx_above50, spx_trend, "
                             "spx_move20, comp_above200, breadth); may be given more than once")
    parser.add_argument("--frozen", action="store_true",
                        help=f"The rule frozen for the holdout on {FROZEN_SET}: --pick {FROZEN_PICK} --only "
                             f"{' --only '.join(FROZEN_ONLY)} (README)")
    parser.add_argument("--explore", action="store_true",
                        help="Only the tables for refining the selection: filters, holds and stops, by 5-year block")
    parser.add_argument("--sizes", type=str, default=SIZES,
                        help="Account test: each short is this %% of the account (2,5)")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check on a chart")
    parser.add_argument("--save-events", type=str, default=None, help="Write every event to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = apply_frozen(parser.parse_args())
    if args.explore and args.universe != "dev":
        raise SystemExit("--explore is for Dev only: the holdout is run once, with --frozen.")
    P = collect(args, ["universe"])
    entry, hold = parse_pick(args.pick)
    fees = [float(x) for x in args.borrow_pcts.split(",") if x.strip()]
    fee = args.pick_fee
    if fee not in fees:
        fees.append(fee)
    low = fees[0]

    t0 = time.time()
    cols = (rules.universe_columns(P) + ["open", "high", "low", "close", "volume", "unadj_close", "turnover", "dv50",
                                         "sma200", "sma50", "hi52_c", "bad_bar", "bad_bar_dn", "bars_since_break",
                                         "listed",
                                         "in_sp500", "in_sp400", "in_sp600", "in_r2000"])
    load_from = (pd.Timestamp(args.start) - pd.Timedelta(days=LOAD_PAD_DAYS)) if args.start else None
    try:
        df = datastore.load_panel(cols, start=load_from, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=TAIL_DAYS)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python pipeline/build_data.py, then python pipeline/build_features.py, "
                         f"and try again.")
    if df.empty:
        raise SystemExit("No stocks loaded for this universe and these dates.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy()
    mkt = pd.read_parquet(config.MARKET_FILE, columns=["date", "spx_open", "spx_close", "comp_open", "comp_close"])
    mkt = mkt.sort_values("date").set_index("date")
    market = datastore.load_market(["spx_above200", "spx_above50", "spx_sma50_rising", "comp_above200",
                                    "pct_above200"]).set_index("date")
    spx200 = market["spx_above200"]
    keep_market = None
    if rules.market_columns(P):
        keep_market = rules.market_mask(df["date"], datastore.load_market(rules.market_columns(P)), P)
    ev, counts = select(df, codes, P, args, keep_market)

    print("=" * 110)
    print(f"REVERSE-SPLIT SHORT  factor jump {args.min_ratio:g}x+  universe={args.universe}  start={args.start}  "
          f"end={args.end}  market={P['market']}  liq top {P['liq_top_pct']:g}%  "
          f"min price ${P['min_price']:g} printed  "
          f"size={P['size']}  pick {args.pick} at {fee:g}%/yr borrow  ({time.time() - t0:.1f}s)")
    print("=" * 110)
    print("Reverse splits found: " + ", ".join(f"{v:,} {k}" for k, v in counts.items())
          + f". Passed the universe: {len(ev):,}.")
    if ev.empty:
        print("No events.")
        return
    pd.set_option("display.width", 250)
    years = ce.span_years(ev["date"])
    ev = features(ev, df, spx200, market, mkt)
    if args.only:
        ev = keep_only(ev, args.only)
        print(f"Kept {len(ev):,} events with {' and '.join(args.only)}.")
        if ev.empty:
            return
    print(f"{len(ev):,} events, {years:.1f} years ({ev['date'].min():%Y-%m-%d} to {ev['date'].max():%Y-%m-%d}), "
          f"{len(ev) / years:.0f} a year. Median ratio 1-for-{ev['ratio'].median():.1f}; median printed close "
          f"${ev['px_prev'].median():.2f} on day -1 and ${ev['px0'].median():.2f} on day 0. Day 0 is a price break "
          f"for {ev['break0'].mean() * 100:.0f}%; {ev['prior_split'].mean() * 100:.0f}% had another reverse split in "
          f"the past 3 years.")
    per_year = ev.groupby(ev["date"].dt.year).size()
    print("Events a year: " + ", ".join(f"{y} {n}" for y, n in per_year.items()))

    grid = [(e, hh) for e in ENTRIES for hh in HOLDS]
    if (entry, hold) not in grid:
        grid.append((entry, hold))
    if args.frozen:
        grid = [(entry, hold)]          # only the frozen rule: other entries and holds stay unseen on the holdout
    name = f"{entry}:{hold}"
    if args.explore:
        cols_out = {}
        for e, hh in grid:
            cols_out.update(trades(ev, df, mkt, e, hh))
        ev = pd.concat([ev, pd.DataFrame(cols_out)], axis=1)
        pick = ev[ev[f"g_{name}"].notna()].reset_index(drop=True)
        tried = explore(pick, ev, df, mkt, name, fee, grid, args)
        print(f"\n{tried} rows above, each one more try on the same Dev data (the first run had 132). Settle on a "
              f"few filters with a reason behind each and that hold in every block, then freeze the rule and test it "
              f"once on the holdout.")
        if not args.no_log:
            res = {k: (round(float(v), 3) if v is not None and np.isfinite(v) else None)
                   for k, v in stats(pick, name, fee, years).items()}
            research_log.append(
                question="Reverse-split short: which filters, hold and stop?",
                phase="reverse_split", split=args.universe, script="reverse_split_events.py --explore",
                params={"start": args.start, "end": args.end, "pick": args.pick, "pick_fee": fee,
                        "min_ratio": args.min_ratio, "dedupe": args.dedupe, "only": args.only, "market": P["market"],
                        "size": P["size"], "liq_top_pct": P["liq_top_pct"], "min_price": P["min_price"]},
                combinations=tried, runs=1, results=res)
        return
    rd = ev["ratio"].where(ev["ratio"] < 2, ev["ratio"].round())
    common = rd.round(1).value_counts().head(10)
    print("\nRatio (factor jump on day 0; a 1-for-10 is 10). Real reverse splits are mostly 1-for-5 to 1-for-50 "
          "and land on round numbers. Most common: " + ", ".join(f"{k:g} ({v})" for k, v in common.items()))
    print(counts_table(ev["ratio"], RATIO_DIST, lambda x: f"{x:g}").to_string(index=False))
    for label, col, edges in [("day -1", "px_prev", PRE_PRICE_EDGES), ("day 0", "px0", [0, 1, 2, 5, 10, np.inf])]:
        t = counts_table(ev[col], edges, lambda x: f"${x:g}")
        print(f"Printed close on {label}: " + ", ".join(f"{b} {n:,}" for b, n in zip(t["bucket"], t["events"])))
    for line in data_checks(ev, df):
        print(line)

    if args.frozen:
        print("\n(--frozen: no day-by-day table and no other entries or holds, so the days after the cover stay "
              "unseen on the holdout.)")
    else:
        print(f"\nDay by day around day 0 against the S&P, in the short's favour (vs_spx = average move that day, + = "
              f"the stock fell more than the S&P; running = from day 1 on, their total from the day 0 close, the drift "
              f"a short at the close would catch; up to day 0, the total from that day to the day 0 close; short_way_% "
              f"= share moving the short's way; med_vol_x = median volume vs usual (days {VOL_BASE[0]} to "
              f"{VOL_BASE[1]}); under1_% = share printing under $1; delisted_% = share that had stopped trading). "
              f"Price-break days are left out of vs_spx, which leaves out most big split-day moves.")
        prof = profile(ev, df, mkt)
        print(prof[prof["day"].isin(PROFILE_SHOW)].round(2).to_string(index=False))

    cols_out = {}
    for e, hh in grid:
        cols_out.update(trades(ev, df, mkt, e, hh))
    ev = pd.concat([ev, pd.DataFrame(cols_out)], axis=1)
    cost = ce.round_trip_cost()
    print(f"\nEvery entry and hold at {low:g}%/yr borrow (ret = after a {cost:.2f}% round trip and the borrow for each "
          f"trading day held; hedged = vs the S&P; hedged_comp = vs the Nasdaq Composite, neither a real small-cap "
          f"hedge; adv50_% = went {ADVERSE:g}%+ against the short before the cover; be_fee = the break-even fee, "
          f"% a year; delist_% = stopped trading during the hold, covered at the last close). hold = trading days "
          f"from the entry close (open1: from the day 0 close).")
    rows = [{"entry": e, "hold": hh, **stats(ev, f"{e}:{hh}", low, years)} for e, hh in grid]
    show = ["entry", "hold", "trades", "per_yr", "ret", "hedged", "hedged_comp", "med", "win_%", "worst5%", "worst",
            "adv50_%", "t", "t_hedged", "be_fee", "delist_%"]
    print(pd.DataFrame(rows)[show].round(2).to_string(index=False))
    rows = []
    for e, hh in grid:
        rn = f"{e}:{hh}"
        row = {"entry": e, "hold": hh, "trades": int(ev[f"g_{rn}"].notna().sum())}
        for f in sorted(fees):
            row[f"ret@{f:g}"] = ret(ev, rn, f).mean()
        row[f"hedged@{fee:g}"] = ret(ev, rn, fee, "h").mean()
        row[f"t@{fee:g}"] = ce.tstat_by_month(ret(ev, rn, fee), ev["date"])
        row["be_fee"] = breakeven(ev, rn)
        rows.append(row)
    print(f"\nThe fee ladder: the same rows at each borrow fee (% a year), hedged and t at {fee:g}%. Borrow is charged "
          f"from the entry for every trading day held, days / 252. be_fee = the fee at which the row breaks even.")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    pick = ev[ev[f"g_{name}"].notna()].reset_index(drop=True)
    off, kind = ENTRIES[entry]
    opened = (ev["row"] + off <= ev["last"]).to_numpy()
    cover_at = (ev["row"] + (0 if kind == "open" else off) + hold).to_numpy()
    n_end = int((opened & (cover_at > ev["last"].to_numpy()) & ~ev["gone"].to_numpy(dtype=bool)).sum())
    n_dn = int((opened & ev[f"g_{name}"].isna().to_numpy()).sum()) - n_end
    entry_label = {"close0": "the day 0 close", "open1": "the day 1 open", "close5": "the day 5 close",
                   "close20": "the day 20 close"}[entry]
    groups = split_groups(pick, args.min_ratio, P["min_price"])
    n_buckets = sum(len(b) for _, b in groups) - 1
    print(f"\nPICK: short at {entry_label}, cover {hold} trading days later, no stop, judged at {fee:g}%/yr borrow "
          f"({len(pick):,} trades; left out: {n_dn} with a downward price break in the hold, {n_end} still open when "
          f"the data ends). ret@{low:g} is the same at {low:g}%/yr. Split by what was known at the day 0 close. "
          f"These are {n_buckets} buckets, so a few will reach t 2 by chance: one is worth a follow-up only if it "
          f"turns the sign and holds in every 5-year block.")
    print(split_table(pick, name, fee, low, years, groups).round(2).to_string(index=False))
    if args.frozen:
        print(f"\nFILTERS, context set on {FROZEN_SET} and not part of the rule: keeping only the trades that pass "
              f"each one, at {fee:g}%/yr. kept_% = share of the trades kept; ret_dropped = the average of the ones it "
              f"drops; total = trades x ret (% of one position), the profit if every trade is the same size.")
        print(filter_table(pick, name, fee).round(2).to_string(index=False))

    st = stats(pick, name, fee, years)
    stress = st.get("ret", np.nan) - (STRESS_COST - cost)
    gone = pick[pick[f"del_{name}"] == 1]
    lastpx = df["unadj_close"].to_numpy(np.float64)[gone["last"].to_numpy()]
    r_gone = ret(gone, name, fee)
    print(f"\nCost stress: the pick at a {STRESS_COST:g}% round trip instead of {cost:.2f}% (spreads on $1-5 stocks): "
          f"{stress:+.2f}% a trade at {fee:g}%/yr.")
    print(f"Delisting: {len(gone)} of {len(pick)} pick trades ({len(gone) / max(len(pick), 1) * 100:.1f}%) stopped "
          f"trading during the hold and were covered at their last exchange close, averaging "
          f"{r_gone.mean() if len(gone) else np.nan:+.2f}%. {int((lastpx < 1).sum())} last printed under $1 (likely "
          f"to OTC or bankrupt; avg {r_gone[lastpx < 1].mean() if (lastpx < 1).any() else np.nan:+.2f}%), "
          f"{int((lastpx >= 1).sum())} at $1+ (often a buyout; avg "
          f"{r_gone[lastpx >= 1].mean() if (lastpx >= 1).any() else np.nan:+.2f}%). The real price after a move to OTC "
          f"is usually lower, so these shorts are understated.")

    print(f"\nPick by 5-year block (at {fee:g}%/yr):")
    blocks = by_period(pick, name, fee, ce.block_of(pick["date"]))
    print(blocks.round(2).to_string(index=False))
    yr = by_period(pick, name, fee, pick["date"].dt.year)
    print(f"\nPick by year (at {fee:g}%/yr):")
    print(yr.round(2).to_string(index=False))
    n_up, n_yr = int((yr["ret"] > 0).sum()), int(yr["ret"].notna().sum())
    print(f"Positive in {n_up} of {n_yr} years.")
    covid = (pick["date"] >= pd.Timestamp(COVID[0])) & (pick["date"] <= pd.Timestamp(COVID[1]))
    st_ex = stats(pick[~covid.to_numpy()], name, fee, years)
    print(f"Without day 0 from {COVID[0]} to {COVID[1]}: {st_ex['trades']} trades, "
          f"{st_ex.get('ret', np.nan):+.2f}% a trade (t {num(st_ex.get('t')):.2f}), "
          f"hedged {st_ex.get('hedged', np.nan):+.2f}%.")

    col = f"short_{name}"
    pick[col] = ret(pick, name, fee)
    dates = df["date"].to_numpy()
    worst = pick.nsmallest(WORST, col)
    print(f"\nLOSSES AND DRAWDOWN at {fee:g}%/yr. The {len(worst)} worst trades (stock_% = the stock's move from "
          f"entry to cover; short = % of the position after costs; peak_% = the highest high above the entry before "
          f"the cover; gone = stopped trading before the cover):")
    print(pd.DataFrame({"symbol": worst["symbol"].to_numpy(),
                        "day0": pd.to_datetime(worst["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                        "cover_day": pd.to_datetime(dates[worst[f"exit_{name}"].to_numpy()]).strftime("%Y-%m-%d"),
                        "ratio": worst["ratio"].to_numpy(), "px0": worst["px0"].to_numpy(),
                        "stock_%": worst[f"raw_{name}"].to_numpy(), "short": worst[col].to_numpy(),
                        "peak_%": worst[f"peak_{name}"].to_numpy(),
                        "gone": worst[f"del_{name}"].to_numpy() == 1}).round(2).to_string(index=False))

    sizes = [float(x) for x in args.sizes.split(",") if x.strip()]
    summary, yrs, skipped = [], {}, 0
    big = (pick["ratio"] >= BIG_RATIO).to_numpy()
    big_col = f"size_{BIG_RATIO:g}+_%"
    if kind == "close":
        pick[f"short_{off}:{off + hold}"] = pick[col]       # the name lockup_events.account() reads
        for size in sizes:
            curve, skipped = le.account(pick, df, mkt["spx_close"], off, off + hold, size, fee)
            if curve.empty:
                continue
            row, r_y = le.account_summary(curve, size)
            summary.append({"size_%": size, big_col: size, **row})
            yrs[f"account_{size:g}%"] = r_y
            if big.any():
                curve, _ = account_sized(pick, df, mkt["spx_close"], off, off + hold, size, fee,
                                         np.where(big, BIG_SCALE, 1.0))
                summary.append({"size_%": size, big_col: size * BIG_SCALE, **le.account_summary(curve, size)[0]})
    rows = []
    for y, g in pick.groupby(pick["date"].dt.year):
        x = g[col]
        w = g.loc[x.idxmin()]
        rows.append({"year": y, "trades": len(g), "short": x.mean(), "losers": int((x <= 0).sum()),
                     "lost_%_sum": x[x <= 0].sum(), "worst": f"{w['symbol']} {w[col]:.1f}",
                     "2nd_worst": f"{x.nsmallest(2).iloc[-1]:.1f}" if len(x) > 1 else ""})
    by_year = pd.DataFrame(rows).set_index("year")
    for k, v in yrs.items():
        by_year[k] = v.reindex(by_year.index)
    print("\nBy year (day 0's year): losers = losing trades, lost_%_sum = their losses added up (% of one position), "
          "worst = the year's worst trade. account_X% = the account's return that calendar year with each short X% "
          "of the account when it opens, marked at every close.")
    print(by_year.round(2).to_string())
    if summary:
        print(f"\nThe account at {fee:g}%/yr borrow: x_start = end balance / start; max_dd = the worst drop from a "
              f"high, marked at every close (dd_from = the high, dd_low = the bottom); max_open / avg_open = shorts "
              f"open at once; max_short_% = the open shorts' value as % of the account at most. No cap on shorts open"
              + (f"; {skipped} trades off the S&P calendar left out." if skipped else ".")
              + f" {big_col} = the size of the 1-for-{BIG_RATIO:g}+ shorts ({int(big.sum())} of {len(pick)} trades): "
              f"each size also at {BIG_SCALE:g}x for them, added after the holdout run because the biggest spikes "
              f"were big ratios.")
        print(pd.DataFrame(summary).round(2).to_string(index=False))
    elif kind != "close":
        print("\nThe account test needs a close entry (close0 / close5 / close20).")

    if args.sample:
        s = pick.sample(min(args.sample, len(pick)), random_state=0).sort_values("date")
        print(f"\n{len(s)} random pick trades to check on a chart (python tools/show_data.py SYMBOL). On day0 the "
              f"printed close should jump by about the ratio while the adjusted close stays smooth:")
        out = pd.DataFrame({"symbol": s["symbol"].to_numpy(),
                            "day0": pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                            "exit_day": pd.to_datetime(dates[s[f"exit_{name}"].to_numpy()]).strftime("%Y-%m-%d"),
                            "ratio": s["ratio"].to_numpy(), "px_day-1": s["px_prev"].to_numpy(),
                            "px_day0": s["px0"].to_numpy(), "stock_%": s[f"raw_{name}"].to_numpy(),
                            "short": s[col].to_numpy()})
        print(out.round(2).to_string(index=False))

    if args.save_events:
        keep = ["symbol", "date", "ratio", "prior_split", "px_prev", "px0", "dv_m", "index", "run60", "day0", "break0",
                "above200", "spx_above200", "gone"] + [f"{k}_{e}:{hh}" for k in ["g", "y"] for e, hh in grid]
        ev[keep].to_csv(args.save_events, index=False)
        print(f"\nSaved {len(ev)} events to {args.save_events}")

    tries = len(grid) * len(fees) + n_buckets
    blk = dict(zip(blocks["period"], blocks["ret"]))
    checks = [
        (f"1. ret above 0 after costs and {fee:g}%/yr borrow, month-clustered t above 2",
         f"{st.get('ret', np.nan):+.2f}%, t {num(st.get('t')):.2f}",
         st.get("ret", -1) > 0 and (st.get("t") or 0) > 2),
        ("2. positive in more than half the years and in each of the 2015-19, 2020-24 and 2025+ blocks",
         f"{n_up} of {n_yr} years; " + ", ".join(f"{b} {blk.get(b, np.nan):+.2f}" for b in NEED_BLOCKS),
         n_up * 2 > n_yr and all(blk.get(b, -1) > 0 for b in NEED_BLOCKS)),
        ("3. big enough: about 5% a trade, or a smaller edge with about 50+ trades a year",
         f"{st.get('ret', np.nan):+.2f}% a trade, {st['per_yr']:.0f} a year",
         st.get("ret", -1) >= 5 or (st.get("ret", -1) > 0 and st["per_yr"] >= 50)),
        (f"4. still positive without day 0 from {COVID[0]} to {COVID[1]}", f"{st_ex.get('ret', np.nan):+.2f}%",
         st_ex.get("ret", -1) > 0),
    ]
    rule = f"the rule frozen on {FROZEN_SET}" if args.frozen else "the pick"
    only = "".join(f", {spec}" for spec in args.only)
    print(f"\nPASS BAR, set before the first run (README): {rule} ({args.pick}{only}, no stop, $"
          f"{P['min_price']:g} printed, top {P['liq_top_pct']:g}% by $ volume, first reverse split in {args.dedupe} "
          f"bars), all four. This run: {args.universe} from {args.start or 'the first bar'}.")
    for label, value, ok in checks:
        print(f"  {label}: {value}  {'yes' if ok else 'NO'}")
    verdict = "PASSES on this data" if all(ok for _, _, ok in checks) else "does not pass"
    print(f"  -> {verdict}. " + ("A Dev pass is not a result: freeze the rule in the README, then run the holdout once."
                                 if args.universe == "dev" else "This is the one holdout run: no changes after it."))
    if args.frozen:
        na, ra, nb, rb, diff, tw = sizing_check(pick, name, fee)
        ok_size = diff > 0 and tw > 2
        print(f"  Sizing check, set the same day: trades with an earlier reverse split in 3 years beat the rest by "
              f"more than 0 with a Welch t above 2 ({na} trades {ra:+.2f}% against {nb} trades {rb:+.2f}%, difference "
              f"{diff:+.2f}, t {tw:.2f})  {'yes' if ok_size else 'NO'} -> "
              + (f"earlier-split trades get {SIZE_UP:g}x the size." if ok_size else "every trade the same size."))
    if args.frozen:
        tries = 1
        print(f"This run is the frozen rule's test: the pass bar and the sizing check. The {n_buckets} split buckets "
              f"and {len(FILTERS)} filters are context.")
    else:
        print(f"Tries in this run: {len(grid)} entry x hold rows at {len(fees)} fees and {n_buckets} split buckets = "
              f"{tries}. The pick is one of them; read the rest as context.")

    if not args.no_log:
        res = {k: (round(float(v), 3) if v is not None and np.isfinite(v) else None) for k, v in st.items()}
        research_log.append(
            question="Do stocks keep falling after a reverse split, by enough to pay a high borrow fee?",
            phase="reverse_split", split=args.universe, script="reverse_split_events.py",
            params={"start": args.start, "end": args.end, "pick": args.pick, "pick_fee": fee, "fees": fees,
                    "min_ratio": args.min_ratio, "dedupe": args.dedupe, "only": args.only, "market": P["market"],
                    "size": P["size"], "liq_top_pct": P["liq_top_pct"], "min_price": P["min_price"]},
            combinations=tries, runs=1, results=res)


if __name__ == "__main__":
    main()
