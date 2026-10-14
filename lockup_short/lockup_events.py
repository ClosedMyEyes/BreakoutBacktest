"""
Lock-up expiry short — do new listings fall when insiders can start selling?
===========================================================================
Idea #3 in ideas/next_setups.md. After an IPO, insiders and early investors are
usually barred from selling for 180 calendar days (the lock-up). When it ends, a
large block of new supply can hit the market on a date everyone could see coming
from the listing day. Studies of the 1990s and 2000s found a 1-3% drop vs the
market around the expiry, with volume up about 40%. This script checks whether
it is there in our data and big enough to trade after costs.

No setups, stops or position sizing: an event table, like the index-change and
crash studies.

The data has no IPO dates, so the listing day is each stock's first bar (the
first_date in symbols.parquet: the first day of its run on NYSE / Nasdaq / NYSE
American). Left out:
  - stocks whose first bar is within 30 days of the start of the data (their
    history just begins there)
  - returns to an exchange (SYMBOL~2): not a new listing
  - SPAC-like starts (the printed close sits at $9.50-10.50 on 90%+ of the first
    20 bars): a SPAC that later merged keeps its shell history, so its first bar
    is the SPAC's IPO and the merger lock-up comes much later. --spacs keep
    puts them back in
Spin-offs (no lock-up) and uplistings from OTC (old holders, no lock-up) can't be
told apart from IPOs, so they dilute the result. An uplisting from a long OTC run
shows up here because build_data.py drops OTC bars.

  day 0   the first trading day on or after first_date + --days calendar days
          (180 by default; --days 90 tests the shorter lock-ups)
  entry   short at the close of day A, cover at the close of day B. Every A and B
          is known from the listing day, so every window is tradable.
  pick    stated before the first run: short the close of day -5, cover the close
          of day +5 (10 days)

The universe is judged at the close of day -20, before any entry: $5 printed
close, the top 80% by 50-day dollar volume, no recent price break, not pinned.
A stock with fewer than 50 bars (day -20 of a 90-day lock-up) is ranked on its
dollar volume since the first bar instead.
A stock that stops trading inside a window covers at its last close. A window
across a downward price break is left out (an untrusted gain for a short); an
upward one counts as a real loss.

Columns (in %; a short earns minus the stock's move):
  short     -raw minus a round trip and --borrow-pct a year for the days held: what
            an unhedged short makes
  hedged    -(stock minus S&P) minus the same costs: a short hedged with the S&P
  b100      short at 100%/yr borrow (new listings are often hard to borrow)
  med / win_% / t   median and share of winning shorts, month-clustered t of short
  hit25_%   share whose daily high went 25%+ above the entry before the cover

HOW TO RUN:
  python lockup_short/lockup_events.py --start 2010-01-01
  python lockup_short/lockup_events.py --start 2010-01-01 --days 90
  python lockup_short/lockup_events.py --start 2010-01-01 --pick=-3:1 --sample 10   (the = is needed before a minus)
  python lockup_short/lockup_events.py --start 2010-01-01 --spacs keep
  python lockup_short/lockup_events.py --start 2010-01-01 --min-price 1 --liq-top-pct 100   # small caps too
  python lockup_short/lockup_events.py --start 2010-01-01 --pick=-20:10 --min-dv 20        # liquid names only

The pick's table splits it by size and liquidity at the day -20 close (price, 50-day
dollar volume, dollar volume in the listing month as a stand-in for size at the IPO;
the data has no market caps), by the move since listing, and by volume around day 0
after the fact, next to the wider windows -20:0, 0:20 and -20:20.

DOES THE DATE MATTER? repeats the main windows with day 0 at 90 to 365 days after
the first bar. A lock-up effect should stand out at 180 (and 90); the same drift
at every age means young stocks lag in general.

The VOLATILITY tables (after the day-by-day profile) ask whether stocks swing more
around day 0 in either direction: the size of daily moves, ranges and gaps, and
5-day blocks from day -60 to +40, each against the stock's own usual.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crash_rebound"))

import config  # noqa: E402
import datastore  # noqa: E402
import research_log  # noqa: E402
import rules  # noqa: E402
from params import add_param_flags, collect  # noqa: E402
import crash_events as ce  # noqa: E402

LOCKUP_DAYS = 180
ENTRIES = [-20, -10, -5, -3, -1, 0]  # short at the close of this day
EXITS = [0, 1, 3, 5, 10, 20]         # cover at the close of this day
PICK = "-5:5"
PROFILE = range(-15, 21)             # days shown in the day-by-day profile
UNIVERSE_DAY = min(ENTRIES)          # universe and size buckets judged at this day's close
SIDE_WINDOWS = [(-20, 0), (0, 20), (-20, 20)]   # also shown (hedged) in the size table
PRICE_BINS = [0, 2, 5, 10, 20, 50, np.inf]               # printed price, $
DV_BINS = [0, 0.5, 2, 5, 20, 100, np.inf]                # dollar volume, $M a day
RUN_BINS = [-100, -50, -20, 0, 25, 100, np.inf]          # % move since the first close
SPIKE_BINS = [0, 1, 2, 4, np.inf]                        # volume around day 0 vs usual
IPO_BARS = 20                                            # the listing month, for size at listing
AGES = [90, 120, 150, 180, 210, 240, 300, 365]           # day 0 at these calendar days after the first bar
AGE_WINDOWS = [(-20, 0), (-20, 10), (-20, 20), (-5, 5)]  # shown (hedged) in the age table
VOL_BASE = (-60, -21)                # rows for the stock's usual volume
DATA_START_GRACE = 30                # first bar this close to the data start = not a listing
SPAC_BARS, SPAC_LO, SPAC_HI, SPAC_SHARE = 20, 9.5, 10.5, 0.9
HARD_BORROW = 100.0
SQUEEZE = 0.25


def parse_pick(spec):
    """'-5:5' -> (-5, 5). The cover must come after the entry."""
    a, b = (int(x) for x in str(spec).split(":"))
    if b <= a:
        raise ValueError(spec)
    return a, b


def spac_like(unadj, first, n_bars=SPAC_BARS):
    """True for stocks whose printed close sat at $9.50-10.50 on most of the first
    n_bars bars (a SPAC shell's trust value). unadj: printed closes, first: each
    stock's first row (one per stock)."""
    out = np.zeros(len(first), dtype=bool)
    for k, f in enumerate(first):
        x = unadj[f:f + n_bars]
        x = x[np.isfinite(x)]
        if len(x) >= n_bars // 2:
            out[k] = ((x >= SPAC_LO) & (x <= SPAC_HI)).mean() >= SPAC_SHARE
    return out


def find_events(df, codes, days=LOCKUP_DAYS, data_start=None, spacs="drop"):
    """One event per new listing: the row of day 0 (first bar on or after the first
    bar's date + `days` calendar days). Returns (events, counts of what was left out)."""
    n = len(df)
    idx = np.arange(n)
    first = pd.Series(idx).groupby(codes).transform("min").to_numpy()
    last = pd.Series(idx).groupby(codes).transform("max").to_numpy()
    starts = np.flatnonzero(idx == first)
    dates = df["date"].to_numpy()
    sym = df["symbol"].astype(str).to_numpy()
    data_start = pd.Timestamp(dates.min() if data_start is None else data_start)
    left = {"history starts with the data": 0, "returns to an exchange (~2)": 0,
            "SPAC-like start": 0, "stopped trading before day 0": 0}

    at_start = pd.DatetimeIndex(dates[starts]) <= data_start + pd.Timedelta(days=DATA_START_GRACE)
    relist = np.array(["~" in s for s in sym[starts]], dtype=bool)
    left["history starts with the data"] = int(at_start.sum())
    left["returns to an exchange (~2)"] = int((relist & ~at_start).sum())
    keep = ~at_start & ~relist
    spac = spac_like(df["unadj_close"].to_numpy(np.float64), starts)
    if spacs == "drop":
        left["SPAC-like start"] = int((spac & keep).sum())
        keep &= ~spac
    starts, spac = starts[keep], spac[keep]

    rows, firsts, lasts, flag = [], [], [], []
    for s, sp in zip(starts, spac):
        target = dates[s] + np.timedelta64(days, "D")
        e = last[s]
        r = s + int(np.searchsorted(dates[s:e + 1], target))
        if r > e:
            left["stopped trading before day 0"] += 1
            continue
        rows.append(r)
        firsts.append(s)
        lasts.append(e)
        flag.append(sp)
    rows = np.asarray(rows, dtype=np.int64)
    ev = pd.DataFrame({"row": rows, "first": np.asarray(firsts, dtype=np.int64),
                       "last": np.asarray(lasts, dtype=np.int64),
                       "symbol": sym[rows] if len(rows) else [],
                       "listed_on": dates[firsts] if len(rows) else [],
                       "date": dates[rows] if len(rows) else [],
                       "spac_like": np.asarray(flag, dtype=bool)})
    ev["bars_listed"] = ev["row"] - ev["first"]
    return ev, left


def window(ev, df, spx, a, b, borrow_pct):
    """Short at the close of day a, cover at the close of day b (offsets from day 0).
    Returns short_/hedged_/b100_/raw_/hit_ columns named a:b. A stock that stops trading
    covers at its last close; a window across a downward price break is left out."""
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    dn = (df["bad_bar_dn"].fillna(0) > 0).to_numpy()
    cdn = np.cumsum(dn)
    dates = df["date"].to_numpy()
    r, first, last = ev["row"].to_numpy(), ev["first"].to_numpy(), ev["last"].to_numpy()
    i, j = r + a, np.minimum(r + b, last)
    ok = (i >= first) & (i <= last) & (j > i)
    i, j = np.where(ok, i, r), np.where(ok, j, r)
    raw = c[j] / c[i] - 1
    pos = spx.index.get_indexer(pd.DatetimeIndex(dates))
    sc = spx.to_numpy()
    pi, pj = pos[i], pos[j]
    sp = np.where((pi >= 0) & (pj >= 0), sc[np.maximum(pj, 0)] / sc[np.maximum(pi, 0)] - 1, np.nan)
    bad = ~ok | (cdn[j] - cdn[i] > 0)
    raw[bad] = np.nan
    days_held = np.maximum(j - i, 1)
    cost = ce.round_trip_cost()
    name = f"{a}:{b}"
    out = {f"raw_{name}": raw * 100,
           f"short_{name}": -raw * 100 - cost - borrow_pct * days_held / 252,
           f"hedged_{name}": -(raw - sp) * 100 - cost - borrow_pct * days_held / 252,
           f"b100_{name}": -raw * 100 - cost - HARD_BORROW * days_held / 252}
    top = np.full(len(r), np.nan)
    for k in range(1, max(b - a, 1) + 1):
        jj = np.minimum(i + k, last)
        top = np.where(i + k <= j, np.fmax(top, h[jj]), top)
    hit = top / c[i] - 1 >= SQUEEZE
    out[f"hit_{name}"] = np.where(bad, np.nan, hit.astype(float))
    return out


def profile(ev, df, spx):
    """Average day-by-day move vs the S&P around day 0, its running total from the
    close of day PROFILE[0]-1, and the median volume against the stock's usual."""
    c = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    dn = (df["bad_bar_dn"].fillna(0) > 0) | (df["bad_bar"].fillna(0) > 0)
    brk = dn.to_numpy()
    dates = df["date"].to_numpy()
    pos = spx.index.get_indexer(pd.DatetimeIndex(dates))
    sc = spx.to_numpy()
    r, first, last = ev["row"].to_numpy(), ev["first"].to_numpy(), ev["last"].to_numpy()
    lo, hi = r + VOL_BASE[0], r + VOL_BASE[1]
    base = np.full(len(r), np.nan)
    for k in np.flatnonzero(lo >= first):
        base[k] = np.nanmean(vol[lo[k]:hi[k] + 1])
    rows, cum = [], 0.0
    for d in PROFILE:
        j = r + d
        ok = (j - 1 >= first) & (j <= last)
        jj, ii = np.where(ok, j, r), np.where(ok, j - 1, r)
        x = c[jj] / c[ii] - 1
        pi, pj = pos[ii], pos[jj]
        sp = np.where((pi >= 0) & (pj >= 0), sc[np.maximum(pj, 0)] / sc[np.maximum(pi, 0)] - 1, np.nan)
        x = np.where(ok & ~brk[jj], (x - sp) * 100, np.nan)
        avg = np.nanmean(x) if np.isfinite(x).any() else np.nan
        cum += 0.0 if np.isnan(avg) else avg
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = np.where(ok, vol[jj] / base, np.nan)
        rows.append({"day": d, "vs_spx": avg, "running": cum,
                     "up_%": (x[np.isfinite(x)] > 0).mean() * 100 if np.isfinite(x).any() else np.nan,
                     "med_vol_x": np.nanmedian(vr) if np.isfinite(vr).any() else np.nan})
    return pd.DataFrame(rows)


VOL_OFFSETS = np.arange(-60, 41)     # days covered by the volatility check
VOL_DAILY = range(-10, 11)           # days shown one by one
VOL_BLOCK = 5                        # then in 5-day blocks over all of VOL_OFFSETS
BIG_DAY, BIG_BLOCK = 5.0, 15.0       # % moves vs the S&P counted as big


def daily_matrix(ev, df, spx, offsets=VOL_OFFSETS):
    """Per event (rows) and day offset (columns): the day's move vs the S&P, its
    high-low range and its opening gap, all in % of the prior close. NaN outside the
    stock's life and on price-break days."""
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    brk = ((df["bad_bar_dn"].fillna(0) > 0) | (df["bad_bar"].fillna(0) > 0)).to_numpy()
    pos = spx.index.get_indexer(pd.DatetimeIndex(df["date"].to_numpy()))
    sc = spx.to_numpy()
    r, first, last = ev["row"].to_numpy(), ev["first"].to_numpy(), ev["last"].to_numpy()
    shape = (len(r), len(offsets))
    x, rng, gap = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    for k, d in enumerate(offsets):
        j = r + d
        ok = (j - 1 >= first) & (j <= last)
        jj, ii = np.where(ok, j, r), np.where(ok, j - 1, r)
        pi, pj = pos[ii], pos[jj]
        sp = np.where((pi >= 0) & (pj >= 0), sc[np.maximum(pj, 0)] / sc[np.maximum(pi, 0)] - 1, np.nan)
        ok &= ~brk[jj] & (c[ii] > 0)
        x[:, k] = np.where(ok, (c[jj] / c[ii] - 1 - sp) * 100, np.nan)
        rng[:, k] = np.where(ok, (h[jj] - lo[jj]) / c[ii] * 100, np.nan)
        gap[:, k] = np.where(ok, np.abs(o[jj] / c[ii] - 1) * 100, np.nan)
    return x, rng, gap


def volatility(ev, df, spx, offsets=VOL_OFFSETS):
    """Does the stock swing more around day 0, either way? Returns (daily, blocks).
    Each stock's usual is its own average over days VOL_BASE (before the run-up to day 0).
    daily: one row per day in VOL_DAILY. blocks: 5-day blocks over all offsets."""
    x, rng, gap = daily_matrix(ev, df, spx, offsets)
    ax = np.abs(x)
    base = (offsets >= VOL_BASE[0]) & (offsets <= VOL_BASE[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        usual_abs = np.nanmean(ax[:, base], axis=1)
        usual_rng = np.nanmean(rng[:, base], axis=1)
        usual_sd = np.sqrt(np.nanmean(x[:, base] ** 2, axis=1))
    usual_abs[usual_abs <= 0] = np.nan
    usual_rng[usual_rng <= 0] = np.nan
    usual_sd[usual_sd <= 0] = np.nan

    def pct(a):
        a = a[np.isfinite(a)]
        return a.mean() * 100 if len(a) else np.nan

    rows = []
    for d in VOL_DAILY:
        k = int(np.flatnonzero(offsets == d)[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append({"day": d, "abs_move": np.nanmean(ax[:, k]),
                         "move_x": np.nanmean(ax[:, k] / usual_abs),
                         "range": np.nanmean(rng[:, k]),
                         "range_x": np.nanmean(rng[:, k] / usual_rng),
                         "gap": np.nanmean(gap[:, k]),
                         f"big{BIG_DAY:g}_%": pct(np.where(np.isfinite(ax[:, k]), ax[:, k] >= BIG_DAY, np.nan))})
    daily = pd.DataFrame(rows)

    rows = []
    for start in range(int(offsets[0]), int(offsets[-1]) - VOL_BLOCK + 2, VOL_BLOCK):
        cols = (offsets >= start) & (offsets < start + VOL_BLOCK)
        xs = x[:, cols]
        full = np.isfinite(xs).all(axis=1)
        tot = np.where(full, (np.prod(1 + np.nan_to_num(xs) / 100, axis=1) - 1) * 100, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.where(full, np.sqrt(np.nanmean(xs ** 2, axis=1)), np.nan)
            rows.append({"days": f"{start} to {start + VOL_BLOCK - 1}",
                         "events": int(full.sum()),
                         "sd_5d": np.nanstd(tot),
                         "abs_5d": np.nanmean(np.abs(tot)),
                         f"big{BIG_BLOCK:g}_%": pct(np.where(full, np.abs(tot) >= BIG_BLOCK, np.nan)),
                         "vol_x": np.nanmean(sd / usual_sd),
                         "range_x": np.nanmean(np.nanmean(rng[:, cols], axis=1) / usual_rng),
                         "avg_5d": np.nanmean(tot)})
    return daily, pd.DataFrame(rows)


def stats(g, name, years):
    out = {"events": len(g), "per_yr": len(g) / years}
    s = g[f"short_{name}"].dropna()
    if s.empty:
        return out
    out["short"] = s.mean()
    out["hedged"] = g[f"hedged_{name}"].mean()
    out["b100"] = g[f"b100_{name}"].mean()
    out["med"] = s.median()
    out["win_%"] = (s > 0).mean() * 100
    out["worst5%"] = s.quantile(0.05)
    out["hit25_%"] = g[f"hit_{name}"].mean() * 100
    out["t"] = ce.tstat_by_month(g[f"short_{name}"], g["date"])
    out["t_hedged"] = ce.tstat_by_month(g[f"hedged_{name}"], g["date"])
    return out


def turnover_of(df):
    """Daily dollar volume (Norgate's turnover, else close x volume)."""
    c = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    tv = df["turnover"].to_numpy(np.float64) if "turnover" in df else c * vol
    return np.where(np.isfinite(tv), tv, c * vol)


def young_dv(df, rows, first):
    """Average dollar volume over the last 50 bars, or since the first bar when the
    stock is younger than that (dv50 needs 50 bars). $ a day, for each row."""
    tv = turnover_of(df)
    out = np.full(len(rows), np.nan)
    for k, (r, f) in enumerate(zip(rows, first)):
        out[k] = np.nanmean(tv[max(f, r - 49):r + 1])
    return out


_CROSS = {}


def fill_young_liquidity(df, rows, first):
    """dv_pct for stocks too young for a 50-day dollar volume (fewer than 50 bars):
    their average dollar volume since the first bar, ranked against the 50-day dollar
    volume of the listed stocks loaded that day (Dev only for a Dev run: half the
    market, picked at random, so the same spread). Returns a copy of dv_pct."""
    dv_pct = df["dv_pct"].to_numpy(np.float64).copy()
    need = np.isnan(dv_pct[rows])
    if not need.any():
        return dv_pct
    rows, first = rows[need], first[need]
    mine = young_dv(df, rows, first)
    d = df["date"].to_numpy()
    if _CROSS.get("df") is not df:     # the day's spread of dv50, sorted once per panel
        dv50 = df["dv50"].to_numpy(np.float64)
        ok = np.isfinite(dv50) & (df["listed"].to_numpy() == 1)
        cs = pd.DataFrame({"date": d[ok], "dv": dv50[ok]}).sort_values(["date", "dv"])
        _CROSS.update(df=df, cd=cs["date"].to_numpy(), cv=cs["dv"].to_numpy())
    cd, cv = _CROSS["cd"], _CROSS["cv"]
    for r, m in zip(rows, mine):
        lo, hi = np.searchsorted(cd, d[r], "left"), np.searchsorted(cd, d[r], "right")
        if hi > lo and np.isfinite(m):
            dv_pct[r] = np.searchsorted(cv[lo:hi], m, "right") / (hi - lo) * 100
    return dv_pct


def select_events(df, codes, P, days, data_start, spacs, start, end, keep_market=None, min_dv=0.0):
    """Events for one lock-up length, inside the dates and passing the universe at
    the day UNIVERSE_DAY close (and, with min_dv, a 50-day dollar volume of at least
    min_dv $M a day there). Returns (events, left-out counts, found, inside dates)."""
    ev, left = find_events(df, codes, days, data_start, spacs)
    n_found = len(ev)
    if start:
        ev = ev[ev["date"] >= pd.Timestamp(start)]
    if end:
        ev = ev[ev["date"] <= pd.Timestamp(end)]
    ev = ev.reset_index(drop=True)
    n_dates = len(ev)
    u = ev["row"].to_numpy() + UNIVERSE_DAY
    inside = u >= ev["first"].to_numpy()
    u, f = np.where(inside, u, ev["row"].to_numpy()), ev["first"].to_numpy()
    d = df.iloc[np.unique(u)].copy() if len(u) else df.iloc[:0].copy()
    filled = fill_young_liquidity(df, u, f) if len(u) else df["dv_pct"].to_numpy()
    d["dv_pct"] = filled[d.index.to_numpy()]
    ok_rows = pd.Series(rules.universe_mask(d, P), index=d.index)
    if keep_market is not None:
        ok_rows &= keep_market[d.index.to_numpy()]
    ok = inside & ok_rows.reindex(u).fillna(False).to_numpy(dtype=bool)
    if min_dv > 0 and len(u):
        dv = df["dv50"].to_numpy(np.float64)[u]
        young = ~np.isfinite(dv)
        if young.any():
            dv[young] = young_dv(df, u[young], f[young])
        ok &= dv >= min_dv * 1e6
    return ev[ok].reset_index(drop=True), left, n_found, n_dates


def age_table(df, codes, spx, P, args, data_start, keep_market):
    """Is day 180 special? The same windows with day 0 at other ages. If the stock
    lags the S&P just as much at 150 or 240 days, the drift is young stocks lagging
    in general, not the lock-up."""
    ages = sorted(set(AGES + [args.days]))
    rows = []
    for age in ages:
        ev, _, _, _ = select_events(df, codes, P, age, data_start, args.spacs, args.start, args.end, keep_market,
                                    args.min_dv)
        out = {"day0_at": f"{age}d", "events": len(ev)}
        if ev.empty:
            rows.append(out)
            continue
        out["per_yr"] = len(ev) / ce.span_years(ev["date"])
        cols = {}
        for a, b in AGE_WINDOWS:
            cols.update(window(ev, df, spx, a, b, args.borrow_pct))
        g = pd.concat([ev, pd.DataFrame(cols)], axis=1)
        for a, b in AGE_WINDOWS:
            out[f"hedged_{a}:{b}"] = g[f"hedged_{a}:{b}"].mean()
        key = "-20:10"
        out[f"t_hedged_{key}"] = ce.tstat_by_month(g[f"hedged_{key}"], g["date"])
        out[f"short_{key}"] = g[f"short_{key}"].mean()
        vol = df["volume"].to_numpy(np.float64)
        r, f = g["row"].to_numpy(), g["first"].to_numpy()
        base = np.array([np.nanmean(vol[max(fi, ri + VOL_BASE[0]):ri + VOL_BASE[1] + 1])
                         if ri + VOL_BASE[1] >= fi else np.nan for ri, fi in zip(r, f)])
        with np.errstate(invalid="ignore", divide="ignore"):
            out["med_vol_x_day0"] = np.nanmedian(vol[r] / base)
        rows.append(out)
    return pd.DataFrame(rows)


def entry_features(ev, df, spx_above200, a=UNIVERSE_DAY):
    """What is known at the close of day a (day -20 by default, before any entry in
    the grid), for the size table, plus one after-the-fact column (spike_x)."""
    c = df["close"].to_numpy(np.float64)
    u = df["unadj_close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    tv = turnover_of(df)
    r, first, last = ev["row"].to_numpy(), ev["first"].to_numpy(), ev["last"].to_numpy()
    i = np.maximum(r + a, first)
    ev["price"] = u[i]
    ev["since_listing"] = (c[i] / c[first] - 1) * 100
    ev["move20"] = np.where(i - 20 >= first, (c[i] / c[np.maximum(i - 20, 0)] - 1) * 100, np.nan)
    dv = df["dv50"].to_numpy(np.float64)[i]
    young = ~np.isfinite(dv)
    if young.any():
        dv[young] = young_dv(df, i[young], first[young])
    ev["dv50_m"] = dv / 1e6
    ipo_dv, ipo_vol, usual, spike = (np.full(len(r), np.nan) for _ in range(4))
    for k in range(len(r)):
        f = first[k]
        ipo_dv[k] = np.nanmean(tv[f:f + IPO_BARS]) / 1e6
        ipo_vol[k] = np.nanmean(vol[f:f + IPO_BARS])
        lo, hi = r[k] + VOL_BASE[0], r[k] + VOL_BASE[1]
        if lo >= f:
            usual[k] = np.nanmean(vol[lo:hi + 1])
        spike[k] = np.nanmax(vol[max(r[k] - 1, f):min(r[k] + 1, last[k]) + 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["ipo_dv_m"] = ipo_dv
        ev["dried"] = usual / ipo_vol
        ev["spike_x"] = spike / usual
    ev["in_sp"] = (df[["in_sp500", "in_sp400", "in_sp600"]].to_numpy()[i].max(axis=1) > 0)
    d = pd.DatetimeIndex(df["date"].to_numpy()[i])
    ev["spx_above200"] = spx_above200.reindex(d).to_numpy()
    return ev


def bins_of(v, edges, fmt):
    """(label, mask) for each fixed bin [lo, hi) of v. The first bin is 'under hi'
    (its low edge is a floor, 0 or -100%), the last 'lo+'."""
    out = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if k == 0:
            label = f"under {fmt(hi)}"
        elif not np.isfinite(hi):
            label = f"{fmt(lo)}+"
        else:
            label = f"{fmt(lo)} to {fmt(hi)}"
        out.append((label, ((v >= lo) & (v < hi)).to_numpy()))
    return out


def thirds_of(v):
    lo, hi = v.quantile(1 / 3), v.quantile(2 / 3)
    return [(f"low third (< {lo:.2f})", (v < lo).to_numpy()),
            ("middle third", ((v >= lo) & (v < hi)).to_numpy()),
            (f"high third (>= {hi:.2f})", (v >= hi).to_numpy())]


def size_table(g, name, years):
    """The pick and the wider windows (SIDE_WINDOWS, hedged) by size, liquidity,
    run-up and the other things known at the day -20 close."""
    side = [f"{a}:{b}" for a, b in SIDE_WINDOWS]

    def row(group, label, mask):
        s = g.loc[np.asarray(mask, dtype=bool)]
        st = stats(s, name, years)
        out = {"group": group, "bucket": label, "events": st["events"], "per_yr": st["per_yr"]}
        for k in ["short", "hedged", "b100", "win_%", "hit25_%", "t_hedged"]:
            out[k] = st.get(k, np.nan)
        for w in side:
            out[f"hedged_{w}"] = s[f"hedged_{w}"].mean() if len(s) else np.nan
        return out

    money = lambda x: f"${x:g}"
    rows = [row("all", "", np.ones(len(g), dtype=bool))]
    groups = [("price at day -20", bins_of(g["price"], PRICE_BINS, money)),
              ("$ volume, $M/day (50-day)", bins_of(g["dv50_m"], DV_BINS, money)),
              ("$ volume in the listing month, $M/day", bins_of(g["ipo_dv_m"], DV_BINS, money)),
              ("move since the first close %", bins_of(g["since_listing"], RUN_BINS, lambda x: f"{x:g}%")),
              ("20-day move into day -20 %", thirds_of(g["move20"])),
              ("volume now vs the listing month", thirds_of(g["dried"])),
              ("after the fact: volume on days -1 to +1 vs usual",
               bins_of(g["spike_x"], SPIKE_BINS, lambda x: f"{x:g}x")),
              ("market", [("S&P above its 200-day", (g["spx_above200"] == 1).to_numpy()),
                          ("S&P below its 200-day", (g["spx_above200"] == 0).to_numpy()),
                          ("in an S&P index", g["in_sp"].to_numpy())])]
    if g["spac_like"].any():
        groups.append(("start", [("SPAC-like", g["spac_like"].to_numpy()),
                                 ("not SPAC-like", (~g["spac_like"]).to_numpy())]))
    for group, buckets in groups:
        for label, mask in buckets:
            rows.append(row(group, label, mask))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--start", type=str, default=None, help="First day-0 date")
    parser.add_argument("--end", type=str, default=None, help="Last day-0 date")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Only once the rule is frozen.")
    parser.add_argument("--days", type=int, default=LOCKUP_DAYS,
                        help="Lock-up length in calendar days after the first bar (180; try 90)")
    parser.add_argument("--pick", type=str, default=PICK,
                        help="A:B = short the close of day A, cover the close of day B (default -5:5; write --pick=-3:1)")
    parser.add_argument("--spacs", choices=["drop", "keep"], default="drop",
                        help="Leave out stocks whose first bars sit at a SPAC's $10 (default) or keep them")
    parser.add_argument("--min-dv", type=float, default=0.0,
                        help="Only stocks trading at least this many $M a day (50-day average) at the day -20 close")
    parser.add_argument("--borrow-pct", type=float, default=5.0, help="Borrow fee, %% a year")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check on a chart")
    parser.add_argument("--save-events", type=str, default=None, help="Write every event to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    pa_, pb = parse_pick(args.pick)

    t0 = time.time()
    cols = (rules.universe_columns(P) + ["open", "high", "low", "close", "volume", "turnover", "dv50", "bad_bar", "bad_bar_dn",
                                         "in_sp500", "in_sp400", "in_sp600"])
    try:
        # Full history: the first bar decides day 0, however long before --start it was
        df = datastore.load_panel(cols, universe=args.universe, unlock_holdout=args.unlock_holdout)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = df["symbol"].cat.codes.to_numpy()
    market = pd.read_parquet(config.MARKET_FILE, columns=["date", "spx_close"])
    spx = market.sort_values("date").set_index("date")["spx_close"]
    spx200 = datastore.load_market(["spx_above200"]).set_index("date")["spx_above200"]
    data_start = pd.read_parquet(config.PRICES_FILE, columns=["date"])["date"].min()

    keep_market = None
    if rules.market_columns(P):
        keep_market = rules.market_mask(df["date"], datastore.load_market(rules.market_columns(P)), P)
    ev, left, n_found, n_dates = select_events(df, codes, P, args.days, data_start, args.spacs,
                                               args.start, args.end, keep_market, args.min_dv)

    print("=" * 110)
    print(f"LOCK-UP EXPIRY SHORT  day 0 = first bar + {args.days} calendar days  universe={args.universe}  "
          f"start={args.start}  end={args.end}  market={P['market']}  liq top {P['liq_top_pct']:g}%  "
          f"min price ${P['min_price']:g}  min $ volume {args.min_dv:g}M/day  borrow {args.borrow_pct:g}%/yr  "
          f"({time.time() - t0:.1f}s)")
    print("=" * 110)
    print(f"New listings found (all years): {n_found:,}. Left out before that: "
          + ", ".join(f"{v:,} {k}" for k, v in left.items()) + ".")
    print(f"Day 0 inside the dates: {n_dates:,}. Passed the universe at the day {UNIVERSE_DAY} close: "
          f"{len(ev):,}.")
    if ev.empty:
        print("No events.")
        return
    years = ce.span_years(ev["date"])
    print(f"{len(ev):,} events, {years:.1f} years ({ev['date'].min():%Y-%m-%d} to {ev['date'].max():%Y-%m-%d}), "
          f"{len(ev) / years:.1f} a year. Median bars from the first bar to day 0: "
          f"{ev['bars_listed'].median():.0f}.")
    pd.set_option("display.width", 250)

    print(f"\nDay by day around day 0 (vs_spx = average move vs the S&P that day, %; running = their total; "
          f"med_vol_x = median volume vs the stock's usual, days {VOL_BASE[0]} to {VOL_BASE[1]}).")
    print("A volume bump near day 0 means the date lines up with real lock-up selling.")
    print(profile(ev, df, spx).round(2).to_string(index=False))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # days before a young stock's first bar
        vd, vb = volatility(ev, df, spx)
    print(f"\nVOLATILITY around day 0: does the stock swing more, either way? All in % vs the S&P. _x columns "
          f"= against the stock's own usual (days {VOL_BASE[0]} to {VOL_BASE[1]}), about 1 when nothing changes "
          f"and 1.3 for 30% bigger. New listings calm down as they age, so compare day 0 with the days either side.")
    print(f"  abs_move = average size of the day's move; range = high-low; gap = size of the opening gap; "
          f"big{BIG_DAY:g}_% = share of days moving {BIG_DAY:g}%+ either way.")
    print(vd.round(2).to_string(index=False))
    print(f"\n  5-day blocks: sd_5d / abs_5d = spread and average size of the 5-day move across stocks; "
          f"big{BIG_BLOCK:g}_% = share moving {BIG_BLOCK:g}%+ either way; vol_x = daily volatility (root mean square move) vs usual; "
          f"avg_5d = average move (direction).")
    print(vb.round(2).to_string(index=False))

    grid = [(a, b) for a in ENTRIES for b in EXITS if b > a]
    if (pa_, pb) not in grid:
        grid.append((pa_, pb))
    cols_out = {}
    for a, b in grid:
        cols_out.update(window(ev, df, spx, a, b, args.borrow_pct))
    ev = pd.concat([ev, pd.DataFrame(cols_out)], axis=1)
    rows = [{"short_at": a, "cover_at": b, "days": b - a, **stats(ev, f"{a}:{b}", years)} for a, b in grid]
    cost = ce.round_trip_cost()
    print(f"\nEvery entry and cover day (short = unhedged short after a {cost:.2f}% round trip and "
          f"{args.borrow_pct:g}%/yr borrow; hedged = vs the S&P; b100 = at 100%/yr borrow):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print(f"\nDOES THE DATE MATTER? The same windows (hedged = short vs the S&P after costs) with day 0 at other "
          f"ages. If stocks lag as much at 150 or 240 days as at 180, the drift is young stocks lagging in "
          f"general, not lock-up selling. med_vol_x_day0 = median volume on day 0 vs usual: a lock-up date should "
          f"show a bump at 180 (and 90) only.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(age_table(df, codes, spx, P, args, data_start, keep_market).round(2).to_string(index=False))

    name = f"{pa_}:{pb}"
    ev = entry_features(ev, df, spx200)
    pick = ev[ev[f"short_{name}"].notna()].reset_index(drop=True)
    print(f"\nPICK: short the close of day {pa_}, cover the close of day {pb} ({len(pick)} trades), by what was "
          f"known at the day {UNIVERSE_DAY} close. short / hedged / b100 / win / hit25 / t_hedged are the pick; "
          f"the hedged_A:B columns are the stock vs the S&P (short side, after costs) over the whole run-up "
          f"(-20:0), the drift after (0:20) and both (-20:20). The listing month's $ volume stands in for size "
          f"at the IPO (the data has no market caps). 'After the fact' rows use volume around day 0, which isn't "
          f"known at entry: they show whether heavy selling days fall more, not a filter.")
    print(size_table(pick, name, years).round(2).to_string(index=False))

    pick["block"] = ce.block_of(pick["date"])
    show = ["events", "short", "hedged", "b100", "med", "win_%", "hit25_%", "t"]
    rows = [{"block": b, **{k: v for k, v in stats(g, name, ce.span_years(g["date"])).items() if k in show}}
            for b, g in pick.groupby("block")]
    print("\nPick by 5-year block:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    rows = [{"year": y, **{k: v for k, v in stats(g, name, 1.0).items() if k in show}}
            for y, g in pick.groupby(pick["date"].dt.year)]
    yr = pd.DataFrame(rows)
    print("\nPick by year:")
    print(yr.round(2).to_string(index=False))
    up = int((yr["short"] > 0).sum())
    print(f"Short positive in {up} of {len(yr)} years; hedged positive in {int((yr['hedged'] > 0).sum())}.")

    if args.sample:
        s = pick.sample(min(args.sample, len(pick)), random_state=0).sort_values("date")
        dates = df["date"].to_numpy()
        s = s.assign(short_day=dates[s["row"] + pa_], cover_day=dates[np.minimum(s["row"] + pb, s["last"])])
        print(f"\n{len(s)} random pick trades to check on a chart (listed_on = first bar):")
        s = s[["symbol", "listed_on", "date", "short_day", "cover_day", "price", "since_listing",
               f"raw_{name}", f"short_{name}"]]
        print(s.round({"price": 2, "since_listing": 2, f"raw_{name}": 2, f"short_{name}": 2}).to_string(index=False))

    if args.save_events:
        keep_cols = ["symbol", "listed_on", "date", "bars_listed", "spac_like", "price", "since_listing",
                     "move20", "dv50_m", "ipo_dv_m", "dried", "spike_x"] + [f"short_{a}:{b}" for a, b in grid] + [f"hedged_{name}"]
        ev[keep_cols].to_csv(args.save_events, index=False)
        print(f"\nSaved {len(ev)} events to {args.save_events}")

    print(f"\nWorth building on, set before the first run: the pick's short (or hedged) return clearly above 0 "
          f"after costs, month-clustered t above 2, positive in more than half the years, and big enough to "
          f"matter (Mark's ballpark: about 5% a trade, or a smaller edge with about 50+ trades a year). "
          f"Grid rows other than the pick are {len(grid) - 1} more tries, so read them as context.")

    if not args.no_log:
        res = {k: (round(float(v), 3) if v is not None and np.isfinite(v) else None)
               for k, v in stats(pick, name, years).items()}
        research_log.append(
            question="Do new listings fall around the 180-day lock-up expiry?",
            phase="lockup", split=args.universe, script="lockup_events.py",
            params={"start": args.start, "end": args.end, "days": args.days, "pick": args.pick,
                    "spacs": args.spacs, "market": P["market"], "liq_top_pct": P["liq_top_pct"],
                    "min_price": P["min_price"], "min_dv": args.min_dv, "borrow_pct": args.borrow_pct},
            combinations=len(grid), runs=1, results=res)


if __name__ == "__main__":
    main()
