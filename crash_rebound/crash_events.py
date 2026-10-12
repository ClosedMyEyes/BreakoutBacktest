"""
Crash rebound — do stocks bounce after a big fall?
==================================================
Idea #5 from the setup shortlist: the mirror of the spike short. After a stock
falls 20 / 30 / 50% or more in 1 or 3 days, forced and panic selling may push
it too far, and a buyer at the close may earn a bounce. The stretch study found
that ordinary dips (the bottom 10% each day) no longer bounce after 2010; it
never tested falls this big.

No stops or exits yet: an event table like the spike table.

Event = the first day a stock's 1-day or 3-day close-to-close fall reaches X%,
with no such fall in the 10 bars before (one event per crash). It must pass the
universe filter on that day ($5 printed price, top 80% by dollar volume, no
recent price break, not pinned at a buyout price).

Returns are RAW (what a long earns before costs), from the event day's close
(--entry next_open: the next day's open) to the close H days later:
  rawH       average % return over H days
  vs_spx10   the same minus the S&P 500 over the same days
  med10      median 10-day return; up10_% = share that rose
  worst5%    the 10-day return 1 in 20 events does worse than
  net10      raw10 minus the round-trip cost
  t10        month-clustered t-stat of raw10 (crashes come in waves)
  delist_%   share that stopped trading within 10 days (their last close is used,
             which flatters a bankruptcy: the real exit is often lower)

Price breaks, long view: a window across an upward break (an untrusted jump)
is left out; a downward one counts as a real loss. A 90%+ one-day fall counts
as a break, so those crash days are left out by the universe filter.

The pick (--pick, default 3:30 = a 30%+ fall over 3 days) then gets tables split
by what was known at the entry close, by 5-year block and by year. One split is
"dropped from an S&P index in the last 20 bars", the forced-selling case the
index-change study found (S&P 600 drops bounced about 3% in 10 days).

HOW TO RUN:
  python crash_rebound/crash_events.py --start 2010-01-01
  python crash_rebound/crash_events.py --start 2010-01-01 --pick 1:20
  python crash_rebound/crash_events.py --start 2010-01-01 --entry next_open
  python crash_rebound/crash_events.py --start 2010-01-01 --sample 10      # events to check on a chart
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Kept in its own folder (a separate idea); shares the repo's data loading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import datastore
import research_log
import rules
from params import add_param_flags, collect

DAYS = [1, 3]
FALLS = [20, 30, 50]
HORIZONS = [1, 3, 5, 10, 20]
MAIN_H = 10
GAP_BARS = 10          # one event per crash: no fall of the same kind in the 10 bars before
INDEX_DROP_BARS = 20   # "dropped from an S&P index" = its index went down within this many bars
VOL_BASE = (-25, -6)


def shift(arr, k, codes):
    """arr shifted by k rows (k > 0 = earlier rows), NaN where it would cross a symbol."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = arr[:-k]
        out[k:][codes[k:] != codes[:-k]] = np.nan
    return out


def shift_bool(arr, k, codes):
    out = np.zeros(len(arr), dtype=bool)
    out[k:] = arr[:-k] & (codes[k:] == codes[:-k])
    return out


def fall_pct(c, codes, days):
    """% close-to-close move over `days` bars (negative = a fall)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c / shift(c, days, codes) - 1) * 100


def fall_events(c, codes, days, thr, gap_bars=GAP_BARS):
    """True on the first day the `days`-bar fall reaches thr %, with no such day in
    the gap_bars bars before."""
    hit = fall_pct(c, codes, days) <= -thr
    seen = np.zeros(len(c), dtype=bool)
    for k in range(1, gap_bars + 1):
        seen |= shift_bool(hit, k, codes)
    return hit & ~seen


def forward(df, codes, rows, market, entry="close"):
    """Raw and vs-S&P forward returns (fractions) for event rows, long view."""
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    idx = np.arange(len(df))
    last = pd.Series(idx).groupby(codes).transform("max").to_numpy()[rows]
    up = (df["bad_bar"].fillna(0) > 0).to_numpy()
    cu = np.cumsum(up)
    dates = df["date"].to_numpy()
    m = market.sort_values("date").set_index("date")
    pos = lambda d: m.index.get_indexer(pd.DatetimeIndex(d))
    spx_c, spx_o = m["spx_close"].to_numpy(), m["spx_open"].to_numpy()

    if entry == "close":
        px = c[rows]
        has_entry = np.ones(len(rows), dtype=bool)
        p0 = pos(dates[rows])
        spx0 = np.where(p0 >= 0, spx_c[p0], np.nan)
    else:
        nxt = np.minimum(rows + 1, last)
        has_entry = nxt > rows
        px = np.where(has_entry, o[nxt], np.nan)
        p0 = pos(dates[nxt])
        spx0 = np.where(has_entry & (p0 >= 0), spx_o[p0], np.nan)
    out = {}
    for h in HORIZONS:
        j = np.minimum(rows + h, last)
        r = c[j] / px - 1
        r[~has_entry | (j == rows) | (cu[j] - cu[rows] > 0)] = np.nan
        if entry != "close":
            r[j < rows + 1] = np.nan
        pj = pos(dates[j])
        sp = np.where(pj >= 0, spx_c[pj], np.nan) / spx0 - 1
        out[f"r{h}"] = r
        out[f"x{h}"] = r - sp
    out["delisted"] = rows + MAIN_H > last
    return out


def tstat_by_month(x, dates):
    x = pd.Series(np.asarray(x, dtype=np.float64), index=pd.to_datetime(np.asarray(dates))).dropna()
    if len(x) < 10:
        return None
    e = x - x.mean()
    g = e.groupby(x.index.to_period("M")).sum()
    if len(g) < 5:
        return None
    se = np.sqrt((g ** 2).sum() * len(g) / (len(g) - 1)) / len(x)
    return float(x.mean() / se) if se > 0 else None


def round_trip_cost():
    """% of price: slippage both sides plus commission at about a $10 stock."""
    return 2 * config.DEFAULT_SLIPPAGE_PCT + 2 * config.COMMISSION_PER_SHARE / 10 * 100


def stats(g, years):
    out = {"events": len(g), "per_yr": len(g) / years}
    if len(g) == 0:
        return out
    for h in HORIZONS:
        out[f"raw{h}"] = g[f"r{h}"].mean() * 100
    r = g[f"r{MAIN_H}"].dropna()
    out[f"vs_spx{MAIN_H}"] = g[f"x{MAIN_H}"].mean() * 100
    out[f"med{MAIN_H}"] = r.median() * 100
    out[f"up{MAIN_H}_%"] = (r > 0).mean() * 100
    out["worst5%"] = r.quantile(0.05) * 100
    out[f"net{MAIN_H}"] = out[f"raw{MAIN_H}"] - round_trip_cost()
    out[f"t{MAIN_H}"] = tstat_by_month(g[f"r{MAIN_H}"], g["date"])
    out["delist_%"] = g["delisted"].mean() * 100
    return out


def span_years(dates):
    d = pd.to_datetime(pd.Series(dates))
    return max(0.1, (d.max() - d.min()).days / 365.25) if len(d) else 0.1


def index_state(df):
    s = np.zeros(len(df), dtype=np.int8)
    s[df["in_sp600"].to_numpy() > 0] = 1
    s[df["in_sp400"].to_numpy() > 0] = 2
    s[df["in_sp500"].to_numpy() > 0] = 3
    return s


def add_splits(ev, df, codes, rows, days):
    """What was known at the event day's close, for the split table."""
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    sma = df["sma200"].to_numpy(np.float64)
    first = pd.Series(np.arange(len(df))).groupby(codes).transform("min").to_numpy()[rows]

    start = rows - days                       # the close the fall is measured from
    ok = start >= first
    st = np.where(ok, start, rows)
    ev["above200_before"] = np.where(ok & ~np.isnan(sma[st]), c[st] >= sma[st], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["ibs"] = np.where(h[rows] > lo[rows], (c[rows] - lo[rows]) / (h[rows] - lo[rows]), np.nan)
        prev = np.where(rows - 1 >= first, c[np.maximum(rows - 1, 0)], np.nan)
        ev["gap_pct"] = (o[rows] / prev - 1) * 100
        b60 = rows - days - 60
        ev["run60"] = np.where(b60 >= first, (c[st] / c[np.maximum(b60, 0)] - 1) * 100, np.nan)
    lo_i, hi_i = rows + VOL_BASE[0], rows + VOL_BASE[1]
    base = np.full(len(rows), np.nan)
    for k in np.flatnonzero(lo_i >= first):
        base[k] = np.nanmean(vol[lo_i[k]:hi_i[k] + 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["vol_ratio"] = vol[rows] / base

    s = index_state(df)
    ev["in_index"] = s[rows] > 0
    dropped = np.zeros(len(rows), dtype=bool)
    for k in range(0, INDEX_DROP_BARS):
        a, b = rows - k, rows - k - 1
        okk = b >= first
        dropped |= okk & (s[np.maximum(a, 0)] < s[np.maximum(b, 0)])
    ev["index_drop"] = dropped
    return ev


SPLITS = [
    ("all", lambda e: np.ones(len(e), dtype=bool)),
    ("stock above its 200d before the fall", lambda e: e["above200_before"] == 1),
    ("stock below its 200d before the fall", lambda e: e["above200_before"] == 0),
    ("S&P above its 200d", lambda e: e["spx_above200"] == 1),
    ("S&P below its 200d", lambda e: e["spx_above200"] != 1),
    ("in an S&P index", lambda e: e["in_index"]),
    ("in no S&P index", lambda e: ~e["in_index"]),
    ("dropped from an S&P index in last 20 bars", lambda e: e["index_drop"]),
    ("event day gapped down 10%+", lambda e: e["gap_pct"] <= -10),
    ("event day gapped down < 10%", lambda e: e["gap_pct"] > -10),
    ("closed in bottom third of day's range", lambda e: e["ibs"] < 1 / 3),
    ("closed in middle third", lambda e: (e["ibs"] >= 1 / 3) & (e["ibs"] < 2 / 3)),
    ("closed in top third (off the lows)", lambda e: e["ibs"] >= 2 / 3),
    ("price $5-10", lambda e: e["price"] < 10),
    ("price $10-30", lambda e: (e["price"] >= 10) & (e["price"] < 30)),
    ("price $30+", lambda e: e["price"] >= 30),
]


def thirds(ev, col, label):
    v = ev[col]
    lo, hi = v.quantile(1 / 3), v.quantile(2 / 3)
    return [(f"{label} low third (< {lo:.1f})", lambda e: e[col] < lo),
            (f"{label} middle third", lambda e: (e[col] >= lo) & (e[col] < hi)),
            (f"{label} high third (>= {hi:.1f})", lambda e: e[col] >= hi)]


def block_of(dates):
    y = pd.to_datetime(pd.Series(dates)).dt.year // 5 * 5
    return (y.astype(str) + "-" + (y + 4).astype(str)).to_numpy()


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
    parser.add_argument("--pick", type=str, default="3:30",
                        help="days:fall%% for the split, block and year tables, e.g. 1:20")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick events to check on a chart")
    parser.add_argument("--save-events", type=str, default=None, help="Write the pick's events to this CSV")
    parser.add_argument("--no-log", action="store_true", help="Don't add a row to the research log")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    pick_days, pick_fall = (float(x) for x in args.pick.split(":"))
    pick_days = int(pick_days)

    t0 = time.time()
    cols = (rules.universe_columns(P) + ["open", "high", "low", "close", "volume", "sma200",
                                         "bad_bar", "in_sp500", "in_sp400", "in_sp600"])
    try:
        # 150 extra calendar days before --start for the 60-day and volume look-backs
        lead = None if args.start is None else pd.Timestamp(args.start) - pd.Timedelta(days=150)
        df = datastore.load_panel(cols, start=lead, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=40)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
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

    def events_for(days, fall):
        rows = np.flatnonzero(fall_events(c, codes, days, fall) & keep)
        ev = pd.DataFrame({"row": rows, "symbol": df["symbol"].to_numpy()[rows],
                           "date": df["date"].to_numpy()[rows],
                           "price": df["unadj_close"].to_numpy(np.float64)[rows],
                           "fall": fall_pct(c, codes, days)[rows]})
        for k, v in forward(df, codes, rows, market, args.entry).items():
            ev[k] = v
        return ev

    all_ev = {(d, f): events_for(d, f) for d in DAYS for f in FALLS}
    if (pick_days, pick_fall) not in all_ev:
        all_ev[(pick_days, pick_fall)] = events_for(pick_days, pick_fall)
    pick = all_ev[(pick_days, pick_fall)]
    years = span_years(df.loc[keep, "date"]) if keep.any() else 0.1

    print("=" * 100)
    print(f"CRASH REBOUND  entry={args.entry}  universe={args.universe}  start={args.start}  end={args.end}  "
          f"liq top {P['liq_top_pct']:g}%  min price ${P['min_price']:g}  ({time.time() - t0:.1f}s)")
    print("=" * 100)
    pd.set_option("display.width", 250)
    cost = round_trip_cost()
    rows = [{"days": d, "fall>=": f, **stats(all_ev[(d, f)], years)} for d in DAYS for f in FALLS]
    print(f"\nFirst day a stock is down X% or more over 1 or 3 days (one event per crash). Raw % returns of a "
          f"long from the {'event close' if args.entry == 'close' else 'next open'}; net{MAIN_H} = raw{MAIN_H} "
          f"minus a {cost:.2f}% round trip.")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if pick.empty:
        print(f"\nNo events for the pick {args.pick}.")
        return
    pick = pick.merge(datastore.load_market(["spx_above200"]), on="date", how="left")
    pick = add_splits(pick, df, codes, pick["row"].to_numpy(), pick_days)
    pick["drop"] = -pick["fall"]
    splits = SPLITS + thirds(pick, "vol_ratio", "event-day volume vs usual") + \
        thirds(pick, "run60", "60-day move before the fall") + thirds(pick, "drop", "size of the fall %")
    rows = [{"split": name, **stats(pick.loc[np.asarray(fn(pick), dtype=bool)], years)}
            for name, fn in splits]
    show = ["split", "events", "per_yr", "raw5", f"raw{MAIN_H}", "raw20", f"med{MAIN_H}", f"up{MAIN_H}_%",
            "worst5%", f"net{MAIN_H}", f"t{MAIN_H}", "delist_%"]
    print(f"\nPick {pick_days}-day fall of {pick_fall:g}%+, split by what was known at the entry:")
    print(pd.DataFrame(rows)[show].round(2).to_string(index=False))

    pick["block"] = block_of(pick["date"])
    rows = [{"block": b, **stats(g, span_years(g["date"]))} for b, g in pick.groupby("block")]
    print(f"\nPick by 5-year block:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    rows = []
    for y, g in pick.groupby(pick["date"].dt.year):
        s = stats(g, 1.0)
        rows.append({"year": y, **{k: s.get(k) for k in ["events", "raw5", f"raw{MAIN_H}", "raw20",
                                                         f"med{MAIN_H}", f"net{MAIN_H}"]},
                     "sum_net": s["events"] * s[f"net{MAIN_H}"] / 100 if s["events"] else 0.0})
    print(f"\nPick by year (sum_net = total of the {MAIN_H}-day trades after cost, in position sizes):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.sample:
        s = pick.sample(min(args.sample, len(pick)), random_state=1).sort_values("date")
        s = s.assign(fall=s["fall"].round(1), price=s["price"].round(2),
                     ret10=(s[f"r{MAIN_H}"] * 100).round(1))
        print(f"\n{len(s)} random events to check on a chart (price = printed close on the event day):")
        print(s[["symbol", "date", "fall", "price", "ret10"]].to_string(index=False))

    if args.save_events:
        pick.drop(columns=["row"]).to_csv(args.save_events, index=False)
        print(f"\nSaved {len(pick)} events to {args.save_events}")

    print(f"\nWorth building on = net{MAIN_H} clearly above 0 with t{MAIN_H} above 2, a usable worst5%, "
          f"and positive in the 2010-2014 / 2015-2019 / 2020-2024 blocks.")

    if not args.no_log:
        s = stats(pick, years)
        research_log.append(
            question="Do stocks bounce after a 20-50% fall in 1 or 3 days?",
            phase="crash", split=args.universe, script="crash_rebound/crash_events.py",
            params={**P, "start": args.start, "end": args.end, "entry": args.entry, "pick": args.pick},
            combinations=len(DAYS) * len(FALLS), runs=1,
            results={k: (round(float(v), 3) if v is not None and np.isfinite(v) else None)
                     for k, v in s.items()})


if __name__ == "__main__":
    main()
