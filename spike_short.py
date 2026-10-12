"""
Spike shorts — real trades with stops
=====================================
reversion_test.py found that stocks that just spiked fade: shorting the close of
the first day a stock is 50%+ up in 3 days, or (a small-cap day trader's rule)
the open of the first gap-up day after a big run, made money in most years
since 2010. Its tables had no stop, and about 1 trade in 20 kept squeezing
another 35-50%. This script trades the setups through the short engine
(engine_short.simulate_short) with stops, so the squeezes are capped and every
cost is charged: slippage, commission, borrow and dividends.

Setups (each the first of its kind in 10 bars of a stock):
  run      the close is X%+ above the lowest close of the K bars before it
           (K = 3: a spike; K = 10: a run)
  close    short the close of the first day the run qualifies
  gap      the run qualified at yesterday's close and today opens above
           yesterday's high: short today's open (the gap-up blow-off)

Exit: a stop S% above the short price (filled at the open if the stock gaps
through it; on the gap entry day any high at the stop counts), else cover at the
close `--hold` bars after the entry. A grid of K = 3 / 10, X = 50 / 100%,
close / gap and stops none / 15 / 25 / 40%.

ret = the short's % return after all costs (borrow at --borrow-pct);
ret_hard also charges a hard-to-borrow fee of 50% a year. sum = total of ret in
position sizes (1.0 = one whole position made); max_dd = worst drop in that
running total, trades taken in exit order at equal size.

HOW TO RUN:
  python spike_short.py --start 2010-01-01                   # the grid + by year for the pick
  python spike_short.py --start 2010-01-01 --pick 10:100:gap:25 --sample 10
  python spike_short.py --start 2010-01-01 --hold 5
  python spike_short.py --start 2010-01-01 --mins 30,50 --stops 0,20
  python spike_short.py --start 2010-01-01 --pick 10:100:gap:25 --min-gap 5
  python spike_short.py --start 2010-01-01 --pick 10:100:gap:25 --min-gap 2 --sizes 2.5,5,10

The pick also gets a squeeze table: its trades split by what was known at entry
(gap size, run size, volume, RSI(14), 52-week high, price, trends, liquidity),
to see which buckets hold the squeezes.

--sizes runs the pick through one account: each short is that % of the account
when it opens, at most --max-shorts open at once, marked to market every close.
Every run adds a row to the research log.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import config
import datastore
import engine
import engine_short
import research_log
import rules
from params import add_param_flags, add_run_flags, collect
from reports import tstat_overlap

RUN_BARS = [3, 10]
RUN_MINS = [50, 100]
TRIGGERS = ["close", "gap"]
STOPS = [0, 15, 25, 40]          # % above the short price, 0 = no stop
FIRST_IN = 10                    # a setup counts only if it didn't also fire in the last 10 bars
HARD_BORROW_PCT = 50.0


def shift_bool(arr, k, sym):
    """Boolean arr shifted k rows earlier, False where it would cross a symbol."""
    out = np.zeros(len(arr), dtype=bool)
    out[k:] = arr[:-k] & (sym[k:] == sym[:-k])
    return out


def run_gain(panel, k):
    """% the close is above the lowest close of the k bars before it (NaN early on)."""
    c, sym = panel.c, panel.sym
    low = np.full(len(c), np.nan)
    for j in range(1, k + 1):
        prev = np.full(len(c), np.nan)
        prev[j:] = np.where(sym[j:] == sym[:-j], c[:-j], np.nan)
        low = prev if j == 1 else np.fmin(low, prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c / low - 1) * 100


def spike_candidates(panel, keep, gain, min_gain, trigger, min_gap=0.0):
    """Entries for one setup in engine.find_entries' format.
    gain: run_gain(panel, K); keep: the universe on each row; min_gap: gap entries
    need an open at least this % above the prior day's high."""
    sym, o, h, c = panel.sym, panel.o, panel.h, panel.c
    in_run = gain >= min_gain
    if trigger == "close":
        cand = in_run
    else:
        prev_ok = np.zeros(len(c), dtype=bool)
        prev_ok[1:] = sym[1:] == sym[:-1]
        cand = np.zeros(len(c), dtype=bool)
        cand[1:] = in_run[:-1] & (o[1:] > h[:-1] * (1 + min_gap / 100.0))
        cand &= prev_ok
    seen = np.zeros(len(c), dtype=bool)
    for k in range(1, FIRST_IN + 1):
        seen |= shift_bool(cand, k, sym)
    t = np.flatnonzero(cand & ~seen)
    if trigger == "close":
        t = t[keep[t]]
        sig, entry_row, raw, at_close = t, t, c[t], np.ones(len(t), dtype=bool)
    else:
        t = t[keep[t - 1]]                     # judged at yesterday's close, before the open
        sig, entry_row, raw, at_close = t - 1, t, o[t], np.zeros(len(t), dtype=bool)
    ok = np.isfinite(raw) & (raw > 0)
    return {"signal_row": sig[ok], "entry_row": entry_row[ok], "entry_raw": raw[ok],
            "struct_stop": np.full(int(ok.sum()), np.nan), "at_close": at_close[ok]}


def exit_params(stop, hold, borrow_pct, slippage_pct=None):
    p = {**engine_short.EXIT_DEFAULTS, "exit": "time", "time_bars": hold,
         "max_hold": max(hold, 60), "stop_mode": "pct",
         "stop_pct": stop if stop > 0 else 1e6,              # "no stop" = one that can't be hit
         "max_risk_pct": 1e9, "borrow_pct": borrow_pct}
    if slippage_pct is not None:
        p["slippage_pct"] = slippage_pct
    return p


INTRADAY_STOPS = ("stopped", "stopped (entry day)")     # filled at the stop price, not at an open


def apply_stop_slip(t, pct):
    """Stress: intraday stops fill pct% above the stop (a squeeze or a halt runs
    through it) instead of at it. Gap-through stops already fill at the open."""
    if t.empty or pct <= 0:
        return t
    t = t.copy()
    hit = t["exit_reason"].isin(INTRADAY_STOPS)
    extra = t.loc[hit, "exit"] * pct / 100.0
    t.loc[hit, "exit"] += extra
    t.loc[hit, "ret_pct"] -= extra / t.loc[hit, "entry"] * 100
    return t


def junk_bar(panel, k):
    """engine_short.simulate_short skips a setup whose trigger or entry bar is a
    price break or a dividend gap; the scanner and the skip count use the same test."""
    if panel.col("bad_bar")[k] == 1 or panel.col("bad_bar_dn")[k] == 1:
        return True
    return bool(k > 0 and panel.has_next[k - 1]
                and panel.col("div")[k] > panel.c[k - 1] * config.DIV_MAX_PCT / 100.0)


def summarize(t, years, hold, borrow_pct):
    if t.empty:
        return {"trades": 0}
    days = (t["exit_date"] - t["entry_date"]).dt.days
    r = t["ret_pct"]
    hard = r - max(HARD_BORROW_PCT - borrow_pct, 0.0) * days / 365.0      # at least 50%/yr
    run = t.sort_values("exit_date")["ret_pct"].cumsum() / 100
    by_year = r.groupby(t["entry_date"].dt.year).sum()
    daily = t.groupby("entry_date")["ret_pct"].mean()
    tt, _ = tstat_overlap(daily, hold) if len(daily) > 3 * hold else (None, 0)
    return {
        "trades": len(t), "per_yr": len(t) / years,
        "win_%": (r > 0).mean() * 100, "ret": r.mean(), "median": r.median(),
        "worst_5%": r.quantile(0.05), "worst": r.min(),
        "stopped_%": t["exit_reason"].str.startswith("stopped").mean() * 100,
        "ret_hard": hard.mean(), "sum": r.sum() / 100, "max_dd": (run - run.cummax()).min(),
        "yrs_up": f"{int((by_year > 0).sum())}/{len(by_year)}", "t": tt,
    }


GAP_BINS = [0, 2, 5, 10, 20, np.inf]
PRICE_BINS = [0, 10, 20, 50, np.inf]


def thirds(x):
    """Bucket x into thirds by count, labelled by their ranges ("12-30")."""
    x = pd.Series(x, dtype=float)
    try:
        cut, edges = pd.qcut(x, 3, retbins=True, duplicates="drop")
    except ValueError:                          # too few distinct values
        return pd.Series("all", index=x.index)
    fmt = lambda v: f"{v:.3g}"
    labels = [f"{fmt(a)}-{fmt(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(x, edges, labels=labels, include_lowest=True).astype(str)


def rsi_simple(c, sym, rows, n=14):
    """RSI over the n bars ending at each row (simple averages); NaN without n bars."""
    out = np.full(len(rows), np.nan)
    ok = rows - n >= 0
    r = rows[ok]
    win = r[:, None] - np.arange(n, -1, -1)[None, :]           # n+1 closes, oldest first
    same = (sym[win] == sym[r][:, None]).all(axis=1)
    d = np.diff(c[win], axis=1)
    up, dn = np.clip(d, 0, None).sum(axis=1), np.clip(-d, 0, None).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(up + dn > 0, 100 * up / (up + dn), 50.0)
    out[np.flatnonzero(ok)] = np.where(same, val, np.nan)
    return out


def reverse_splits_before(panel, rows, bars=252):
    """Reverse splits in the `bars` bars up to each row: the printed price jumped
    up against the split-adjusted one (unadj / adjusted factor up over config's
    break factor) on that day. Only splits already done by then count."""
    c, sym = panel.c, panel.sym
    f = panel.col("unadj_close") / c
    rs = np.zeros(len(c), dtype=np.int64)
    rs[1:] = (f[1:] / f[:-1] > config.BREAK_FACTOR_JUMP) & (sym[1:] == sym[:-1])
    cum = np.cumsum(rs)
    starts = np.flatnonzero(np.r_[True, sym[1:] != sym[:-1]])
    first = starts[np.searchsorted(starts, rows, side="right") - 1]
    lo = np.maximum(rows - bars + 1, first)
    return cum[rows] - np.where(lo > 0, cum[np.maximum(lo - 1, 0)], 0)


def entry_features(panel, t, gain, spx_above200):
    """What was known when each trade was entered (all from the signal day, plus the
    gap at the open for gap entries), bucketed for the squeeze table."""
    s, e = t["signal_row"].to_numpy(), t["entry_row"].to_numpy()
    c, o, h = panel.c, panel.o, panel.h
    f = pd.DataFrame(index=t.index)
    gap = np.where(e > s, (o[e] / h[s] - 1) * 100, np.nan)
    if np.isfinite(gap).any():
        f["gap above prior high %"] = pd.cut(gap, GAP_BINS, right=False,
                                             labels=["0-2", "2-5", "5-10", "10-20", "20+"])
    f["run size %"] = thirds(gain[s]).to_numpy()
    f["volume vs 50d avg (x)"] = thirds(panel.col("vol_ratio")[s]).to_numpy()
    f["RSI(14)"] = thirds(rsi_simple(c, panel.sym, s)).to_numpy()
    hi = panel.col("hi52_c")[s]
    f["vs 52-week high"] = np.where(~np.isfinite(hi), "under a year of data",
                                    np.where(c[s] >= hi * 0.999, "new 52w high", "below 52w high"))
    f["share price $"] = pd.cut(panel.col("unadj_close")[s], PRICE_BINS, right=False,
                                labels=["5-10", "10-20", "20-50", "50+"])
    sma = panel.col("sma200")[s]
    f["stock vs its 200d"] = np.where(~np.isfinite(sma), "no 200d yet",
                                      np.where(c[s] >= sma, "above", "below"))
    f["S&P vs its 200d"] = np.where(spx_above200.reindex(panel.date[s]).to_numpy() == 1, "above", "below")
    f["liquidity rank"] = thirds(panel.col("dv_pct")[s]).to_numpy()
    f["reverse split in past year"] = np.where(reverse_splits_before(panel, s) > 0, "yes", "no")
    return f


def squeeze_table(t, feats):
    """Each feature's buckets side by side: which ones hold the squeezes?"""
    rows = []
    r = t["ret_pct"]
    stopped = t["exit_reason"].str.startswith("stopped")
    for col in feats.columns:
        for val, idx in feats.groupby(col, observed=True).groups.items():
            g = r.loc[idx]
            rows.append({"feature": col, "bucket": str(val), "trades": len(g),
                         "win_%": (g > 0).mean() * 100, "ret": g.mean(), "median": g.median(),
                         "worst_5%": g.quantile(0.05), "stopped_%": stopped.loc[idx].mean() * 100})
    return pd.DataFrame(rows)


def skip_report(panel, cand, t, start=None, end=None):
    """Setups the pick did not trade, and why: a junk bar on the trigger or entry
    day (simulate_short skips it), or a trade already open on the stock."""
    d = panel.date[cand["entry_row"]]
    ok = np.ones(len(d), dtype=bool)
    if start:
        ok &= d >= np.datetime64(pd.Timestamp(start))
    if end:
        ok &= d <= np.datetime64(pd.Timestamp(end))
    rows = cand["entry_row"][ok]
    junk = sum(junk_bar(panel, r) or (panel.has_next[r - 1] and junk_bar(panel, r - 1)) for r in rows)
    return (f"\nSetups for the pick: {len(rows)}, traded {len(t)}, skipped for a junk bar (price break or "
            f"dividend gap on the trigger or entry day) {junk}, skipped with a trade already open "
            f"{len(rows) - len(t) - junk}.")


def account_curve(panel, t, dates, size_pct, max_shorts=10, capital=100_000.0, seed=0):
    """Trade the pick through one account. Each short is size_pct% of the account
    (as marked at the prior close) when it opens; at most max_shorts are open at
    once, same-day signals taken in random order. Open shorts are marked at every
    close, and a trade's full ret_pct (costs, borrow and dividends included) is
    booked the day it's covered. Returns the daily curve and the trades skipped."""
    c = panel.c
    di = np.searchsorted(dates, panel.date)
    entries = {}
    for tr in t.sample(frac=1, random_state=seed).itertuples():
        entries.setdefault(di[tr.entry_row], []).append(tr)
    open_, realized, equity, skipped, rows = [], 0.0, capital, 0, []
    for k, d in enumerate(dates):
        for tr in entries.get(k, []):
            if len(open_) >= max_shorts:
                skipped += 1
                continue
            notional = max(equity, 0.0) * size_pct / 100
            open_.append({"row": tr.entry_row, "exit_row": tr.exit_row, "entry": tr.entry,
                          "shares": notional / tr.entry, "pnl": notional * tr.ret_pct / 100})
        unreal, short_value, still = 0.0, 0.0, []
        for pos in open_:
            if di[pos["exit_row"]] <= k:
                realized += pos["pnl"]
                continue
            while pos["row"] < pos["exit_row"] and di[pos["row"] + 1] <= k:
                pos["row"] += 1
            unreal += pos["shares"] * (pos["entry"] - c[pos["row"]])
            short_value += pos["shares"] * c[pos["row"]]
            still.append(pos)
        open_ = still
        equity = capital + realized + unreal
        rows.append({"date": d, "equity": equity, "invested": short_value, "positions": len(open_)})
    return pd.DataFrame(rows), skipped


def account_summary(curve, t, size_pct, skipped, capital=100_000.0):
    m = engine.portfolio_metrics(curve)
    eq = curve.set_index("date")["equity"]
    year_end = eq.groupby(eq.index.year).last()
    by_year = (year_end / year_end.shift(1).fillna(capital) - 1) * 100
    mc = engine.monte_carlo_dd(t["ret_pct"] * size_pct / 100, risk_pct=1.0)
    return {"size_%": size_pct, "cagr": m["cagr"], "max_dd": m["max_dd_pct"], "mar": m["mar"],
            "worst_year": by_year.min(), "longest_dd_days": m["longest_dd_days"],
            "max_open": int(curve["positions"].max()),
            "max_short_%": (curve["invested"] / curve["equity"]).max() * 100,
            "skipped": skipped, "mc_dd_med": mc["mc_dd_p50"], "mc_dd_1in20": mc["mc_dd_p5"]}, by_year


def main():
    parser = argparse.ArgumentParser()
    add_param_flags(parser, ["universe"])
    add_run_flags(parser)
    parser.set_defaults(liq_top_pct=80.0)
    parser.add_argument("--hold", type=int, default=10, help="Bars to hold if not stopped")
    parser.add_argument("--borrow-pct", type=float, default=engine_short.EXIT_DEFAULTS["borrow_pct"],
                        help="Borrow fee, %% a year (ret_hard adds a 50%%/yr stress on top)")
    parser.add_argument("--mins", type=str, default=",".join(str(x) for x in RUN_MINS),
                        help="Run sizes to test, %%, comma separated")
    parser.add_argument("--stops", type=str, default=",".join(str(x) for x in STOPS),
                        help="Stops to test, %% above the short price, 0 = none")
    parser.add_argument("--slippage-pct", type=float, default=None,
                        help="Stress: slippage %% per fill (default %(default)s = the engine's 0.2)")
    parser.add_argument("--stop-slip", type=float, default=0.0,
                        help="Stress: intraday stops fill this %% above the stop instead of at it")
    parser.add_argument("--min-gap", type=float, default=0.0,
                        help="Gap entries: the open must be at least this %% above the prior high")
    parser.add_argument("--pick", type=str, default="3:50:close:25",
                        help="K:X:trigger:stop for the by-year table, e.g. 10:100:gap:25")
    parser.add_argument("--save-trades", type=str, default=None, help="CSV path for the pick's trades")
    parser.add_argument("--sample", type=int, default=0, help="Print N random pick trades to check")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sizes", type=str, default="",
                        help="Account test for the pick: %% of the account per short, comma separated, e.g. 2.5,5,10")
    parser.add_argument("--max-shorts", type=int, default=10, help="Account test: most shorts open at once")
    args = parser.parse_args()
    P = collect(args, ["universe"])
    t0 = time.time()

    cols = (["open", "high", "low", "close", "unadj_close"] + rules.universe_columns(P)
            + engine_short.exit_columns({"exit": "time"}) + ["vol_ratio", "sma200", "hi52_c", "dv_pct"])
    try:
        df = datastore.load_panel(cols, start=args.start, end=args.end, universe=args.universe,
                                  unlock_holdout=args.unlock_holdout, tail_days=120)
    except KeyError as e:
        raise SystemExit(f"{e}. Run python build_data.py, then python build_features.py, and try again.")
    symbols = datastore.load_symbols()
    panel = engine.Panel(df, set(symbols.loc[symbols["delisted"], "symbol"]))
    keep = rules.universe_mask(panel.df, P).copy()
    if rules.market_columns(P):
        keep &= rules.market_mask(panel.df["date"], datastore.load_market(rules.market_columns(P)), P)
    if args.end is not None:
        keep &= (panel.df["date"] <= pd.Timestamp(args.end)).to_numpy()
    d = panel.df.loc[keep, "date"]
    years = max(0.1, (d.max() - d.min()).days / 365.25) if len(d) else 0.1

    pk = args.pick.split(":")
    pick = (int(pk[0]), float(pk[1]), pk[2], float(pk[3]))
    gains = {k: run_gain(panel, k) for k in RUN_BARS + ([pick[0]] if pick[0] not in RUN_BARS else [])}
    rows, pick_trades, pick_cand = [], None, None
    mins = [float(x) for x in args.mins.split(",")]
    stops = [float(x) for x in args.stops.split(",")]
    combos = [(k, x, trg, s) for k in RUN_BARS for x in mins for trg in TRIGGERS for s in stops]
    if pick not in combos:
        combos.append(pick)
    for k, x, trg, s in combos:
        cand = spike_candidates(panel, keep, gains[k], x, trg, args.min_gap)
        t = engine_short.run_signals(panel, cand, exit_params(s, args.hold, args.borrow_pct, args.slippage_pct),
                                     args.start, args.end, family=f"{k}:{x:g}:{trg}:{s:g}")
        t = apply_stop_slip(t, args.stop_slip)
        rows.append({"bars": k, "gain>=": x, "entry": trg, "stop": s or "none",
                     **summarize(t, years, args.hold, args.borrow_pct)})
        if (k, x, trg, s) == pick:
            pick_trades, pick_cand = t, cand

    print("=" * 78)
    print(f"SPIKE SHORTS  hold={args.hold}  universe={args.universe}  start={args.start}  end={args.end}  "
          f"liq_top_pct={P['liq_top_pct']:g}  min_gap={args.min_gap:g}%  ({time.time() - t0:.1f}s)")
    print("=" * 78)
    print("bars/gain>= : the close is gain% above the lowest close of the `bars` bars before it. "
          "entry: close = short that close; gap = short the next open if it is above that day's high.")
    print("ret = % a trade after slippage, commission, borrow and dividends; ret_hard adds 50%/yr borrow; "
          "worst_5% = the 1-in-20 loss; sum / max_dd in position sizes.")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if pick_trades is None or pick_trades.empty:
        print(f"\nNo trades for the pick {args.pick}")
        return
    t = pick_trades
    y = t.groupby(t["entry_date"].dt.year)["ret_pct"]
    tab = pd.DataFrame({"trades": y.size(), "win_%": y.apply(lambda r: (r > 0).mean() * 100),
                        "ret": y.mean(), "worst": y.min(), "sum": y.sum() / 100})
    print(f"\nPick {args.pick} (bars:gain:entry:stop) by year:")
    print(tab.round(2).to_string())
    print("\nExit reasons:")
    g = t.groupby("exit_reason")["ret_pct"]
    print(pd.DataFrame({"trades": g.size(), "ret": g.mean()}).round(2).to_string())
    print(skip_report(panel, pick_cand, t, args.start, args.end))

    spx = datastore.load_market(["spx_above200"])
    spx = pd.Series(spx["spx_above200"].to_numpy(), index=pd.to_datetime(spx["date"]).to_numpy())
    feats = entry_features(panel, t, gains[pick[0]], spx)
    print(f"\nWhat separates the winners from the squeezes? Pick {args.pick}, all {len(t)} trades "
          f"(ret {t['ret_pct'].mean():.2f}%), split by what was known at entry. A bucket with low ret, "
          f"a bad worst_5% and many stops is where the squeezes live. Thirds are by count.")
    print(squeeze_table(t, feats).round(2).to_string(index=False))

    if args.save_trades:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_trades)), exist_ok=True)
        t.to_csv(args.save_trades, index=False)
    if args.sample:
        s = t.sample(min(args.sample, len(t)), random_state=args.sample_seed).sort_values("entry_date")
        p = exit_params(pick[3], args.hold, args.borrow_pct, args.slippage_pct)
        slip, comm = p["slippage_pct"] / 100.0, p["commission"]
        view = pd.DataFrame({
            "symbol": s["symbol"], "shorted": s["entry_date"].dt.date,
            "at": np.where(s["at_close"], "close", "open"),
            "short_px": ((s["entry"] + comm) / (1 - slip)).round(2),
            "real_px": ((s["entry"] + comm) / (1 - slip)
                        * panel.col("unadj_close")[s["entry_row"]] / panel.c[s["entry_row"]]).round(2),
            "stop": s["stop"].round(2) if pick[3] > 0 else np.nan,
            "covered": s["exit_date"].dt.date, "cover_px": ((s["exit"] - comm) / (1 + slip)).round(2),
            "reason": s["exit_reason"], "ret_%": s["ret_pct"].round(1)})
        print(f"\n{len(s)} random trades to check on a chart (prices split-adjusted to today, like "
              f"TradingView; real_px = the price it actually traded at that day; "
              f"python show_data.py SYMBOL --around DATE):")
        print(view.to_string(index=False))

    if args.sizes:
        start = pd.Timestamp(args.start) if args.start else d.min()
        dates = pd.DatetimeIndex(np.unique(panel.date[panel.date >= max(start, d.min())]))
        rows, years_tab = [], {}
        for size in [float(x) for x in args.sizes.split(",")]:
            curve, skipped = account_curve(panel, t, dates, size, args.max_shorts)
            row, by_year = account_summary(curve, t, size, skipped)
            rows.append(row)
            years_tab[f"{size:g}%"] = by_year
        print(f"\nAccount test, pick {args.pick}, {dates[0].date()} to {dates[-1].date()}: each short is size_% of "
              f"the account when it opens (sized on the prior close), at most {args.max_shorts} open at once, "
              f"marked to market at every close. cagr / max_dd / worst_year in % of the account; max_short_% = "
              f"most of the account short at once; skipped = trades with no room; mc_dd = the worst drop if the "
              f"same trades came in a random order (median / 1-in-20).")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        print("\nAccount % return by year (column = % of the account per short):")
        print(pd.DataFrame(years_tab).round(1).to_string())

    research_log.append(
        question=f"Spike shorts, pick {args.pick}, min_gap {args.min_gap:g}%, hold {args.hold}",
        phase="spike", split=args.universe, script="spike_short.py",
        params={**vars(args), **P}, combinations=len(combos), runs=len(combos),
        results={"pick": summarize(t, years, args.hold, args.borrow_pct)})


if __name__ == "__main__":
    main()
