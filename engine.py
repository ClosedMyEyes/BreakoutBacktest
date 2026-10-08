"""
Trade engine: daily-bar fill model, exits, F0 baseline, portfolio, metrics
==========================================================================
Fill model (plan 4.5)
  Signal timing     computed at day t close; orders work on day t+1
  entry_at=close    (default; plan timing b) buy at the close of the first day
                    that closes above the pivot, as you would with a market-on-
                    close order a few minutes before the bell. Skip if that close
                    is more than max_chase % above the pivot. With vol_mult > 0
                    that day's volume must be >= vol_mult x the 50-day average
                    (daily bars show full-day volume, so the volume check is
                    slightly optimistic: see the plan note). F6 buys at the close
                    of the retest day.
  entry_at=stop     (plan timing a) buy stop at the pivot + slippage; if day t+1
                    opens above the pivot, fill at the open + slippage. Skip if the
                    open is more than max_chase % above the pivot.
  entry_at=next_open (plan timing c) same condition as close, buy the next open
  Stop hit          exit at stop - slippage; if it opens below, at open - slippage
  Entry + stop      same day (buy-stop or open fills only): see entry_day_stop
  Stop + target     same day: assume the stop was hit (order unknown)
  Target hit        exit at target - slippage; if it opens above, at open - slippage
  Close-based exits (trailing average, time stop) exit at the next open
  Time exit         exit at the close of bar N
  Delisting         exit at the last traded price; bankruptcies are real losses
  Costs             slippage % per side + commission per share on every fill

Results are in R, where R = entry fill - initial stop level.

Exit models
  x1    fixed stop + target
  x2    O'Neil: stop + target, except if the stock is up fast_gain_pct within
        fast_bars of entry, the target is switched off and the trade is held at
        least hold_bars from entry (initial stop still active), then exits by
        after_hold (trail50 = next open after a close below the 50-day; exit =
        at the close of the last hold bar). hold_stop=breakeven raises the stop
        to the entry price from the bar after the hold starts.
  x3    stop, then exit on a close below the trailing average (trail)
  time  stop (optional), exit at the close after time_bars
  Any model also takes ts_bars/ts_min_gain (X5 time stop) and max_hold.
  stop_mode=structural uses the entry family's stop (X6) instead of stop_pct.
"""

import numpy as np
import pandas as pd

import config

EXIT_DEFAULTS = {
    "exit":          "x2",
    "stop_mode":     "pct",     # pct / structural
    "stop_pct":      7.5,       # % below entry fill (0 = no stop; R uses 7.5% nominal)
    "stop_buffer":   0.0,       # extra % below a structural stop
    "max_risk_pct":  15.0,      # skip setups whose stop is further than this below entry
    "target_pct":    22.5,
    "fast_gain_pct": 20.0,
    "fast_bars":     15,        # 3 weeks
    "hold_bars":     40,        # 8 weeks
    "after_hold":    "trail50",
    "hold_stop":     "initial", # x2 once the hold starts: initial = keep the original stop,
                                # breakeven = raise it to the entry price from the next bar
    "trail":         "sma50",   # sma50 / ema21 / sma10
    "time_bars":     20,
    "ts_bars":       0,         # X5: exit if not up ts_min_gain % after ts_bars (0 = off)
    "ts_min_gain":   5.0,
    "max_hold":      252,
    "partial_frac":  0.33,      # X4: sell this fraction at the first target...
    "partial_r":     2.0,       # ...at +partial_r R
    "partial_pct":   0.0,       # ...or at +partial_pct % when > 0; then stop to breakeven
                                # and trail the rest with `trail` (close below -> next open)
    "entry_day_stop": "path",   # buy-stop filled intraday and the day's low is at the stop:
                                # path = assume a green bar went open-low-high-close (low came
                                # before the fill, not stopped) and a red bar open-high-low-close
                                # (stopped); touch = always stopped (harsh: F0 enters at the
                                # open and never faces this, so touch biases edge_R down)
    "slippage_pct":  config.DEFAULT_SLIPPAGE_PCT,
    "commission":    config.COMMISSION_PER_SHARE,
}

NOMINAL_R_PCT = 7.5


def exit_columns(params):
    p = {**EXIT_DEFAULTS, **{k: v for k, v in params.items() if k in EXIT_DEFAULTS}}
    cols = []
    if p["exit"] in ("x3", "x4"):
        cols.append(p["trail"])
    if p["exit"] == "x2" and p["after_hold"] == "trail50":
        cols.append("sma50")
    return cols


# =============================================================================
# PANEL ARRAYS
# =============================================================================

class Panel:
    """Numpy views of a (symbol, date)-sorted DataFrame for fast trade loops."""

    def __init__(self, df, delisted):
        cat = df["symbol"].astype("category")
        codes = cat.cat.codes.to_numpy()
        if np.any(np.diff(codes) < 0):           # files are already (symbol, date) sorted
            order = np.lexsort((df["date"].to_numpy(), codes))
            df, cat = df.iloc[order], cat.iloc[order]
        df = df.reset_index(drop=True)
        cat = cat.reset_index(drop=True)
        self.df = df
        self.sym = cat.cat.codes.to_numpy().astype(np.int32)    # integer symbol ids
        self.sym_names = np.asarray(cat.cat.categories)
        self.date = df["date"].to_numpy()
        self.o = df["open"].to_numpy(np.float64)
        self.h = df["high"].to_numpy(np.float64)
        self.l = df["low"].to_numpy(np.float64)
        self.c = df["close"].to_numpy(np.float64)
        same_next = np.zeros(len(df), dtype=bool)
        same_next[:-1] = self.sym[1:] == self.sym[:-1]
        self.has_next = same_next                       # row i+1 is the same symbol
        # index of each symbol's last row
        last = np.flatnonzero(~same_next)
        self.last_row = np.repeat(last, np.diff(np.concatenate([[-1], last])))
        self.delisted = cat.isin(list(delisted)).to_numpy()
        self.cols = {}

    def col(self, name):
        if name not in self.cols:
            self.cols[name] = self.df[name].to_numpy(np.float64)
        return self.cols[name]


# =============================================================================
# TRIGGERS: which setups actually fill
# =============================================================================

def find_entries(panel, setups, entry_p):
    """Return candidate entries: arrays of (signal_row, entry_row, entry_raw,
    struct_stop, entry_at_close)."""
    setup, pivot, sstop = setups["setup"], setups["pivot"], setups["stop"]
    o, h, c = panel.o, panel.h, panel.c
    n = len(o)
    t = np.flatnonzero(setup & panel.has_next)
    t1 = t + 1
    piv = pivot[t]
    mc = entry_p["max_chase"] / 100.0
    entry_at = entry_p["entry_at"]
    zeros, ones = np.zeros(len(t), dtype=bool), np.ones(len(t), dtype=bool)

    if setups.get("entry") == "open":              # F6: the signal day itself is the trigger
        ok = ones
        if entry_at == "close":
            entry_row, raw, at_close = t, c[t], ones
        else:
            entry_row, raw, at_close = t1, o[t1], zeros
    elif entry_at == "stop":
        ok = (h[t1] > piv) & (o[t1] <= piv * (1 + mc))
        entry_row, raw, at_close = t1, np.maximum(o[t1], piv), zeros
    elif entry_at in ("close", "next_open"):
        conf = c[t1] > piv
        if entry_p["vol_mult"] > 0:
            conf &= panel.col("vol_ratio")[t1] >= entry_p["vol_mult"]
        if entry_at == "close":
            ok = conf & (c[t1] <= piv * (1 + mc))
            entry_row, raw, at_close = t1, c[t1], ones
        else:
            has2 = np.zeros(len(t), dtype=bool)
            inside = t1 < n
            has2[inside] = panel.has_next[t1[inside]]
            t2 = np.where(has2, t1 + 1, t1)
            ok = conf & has2 & (o[t2] <= piv * (1 + mc))
            entry_row, raw, at_close = t2, o[t2], zeros
    else:
        raise ValueError(f"entry_at must be close, stop or next_open, not {entry_at!r}")
    ok &= np.isfinite(raw)
    return {
        "signal_row": t[ok], "entry_row": entry_row[ok], "entry_raw": raw[ok],
        "struct_stop": sstop[t][ok], "at_close": at_close[ok],
    }


# =============================================================================
# ONE TRADE
# =============================================================================

def simulate_trade(panel, i_entry, entry_raw, at_close, struct_stop, p, risk_pct_override=None):
    """Walk forward from the entry bar until an exit. Returns a dict or None if
    the setup is skipped (stop invalid or too wide)."""
    o, h, l, c = panel.o, panel.h, panel.l, panel.c
    slip = p["slippage_pct"] / 100.0
    comm = p["commission"]

    entry = entry_raw * (1 + slip) + comm
    if risk_pct_override is not None:
        stop = entry * (1 - risk_pct_override)
    elif p["stop_mode"] == "structural":
        if not np.isfinite(struct_stop):
            return None
        stop = struct_stop * (1 - p["stop_buffer"] / 100.0)
    elif p["stop_pct"] > 0:
        stop = entry * (1 - p["stop_pct"] / 100.0)
    else:
        stop = -np.inf
    if np.isfinite(stop):
        if stop >= entry or (entry - stop) / entry > p["max_risk_pct"] / 100.0:
            return None
        R = entry - stop
    else:
        R = entry * NOMINAL_R_PCT / 100.0

    init_stop = stop           # sizing and R always use the initial stop
    model = p["exit"]
    target = entry_raw * (1 + p["target_pct"] / 100.0) if model in ("x1", "x2") else np.inf
    trail = None
    if model in ("x3", "x4"):
        trail = panel.col(p["trail"])
    partial_px = None          # x4: fill price of the partial sale once done
    if model == "x4":
        part_at = entry_raw * (1 + p["partial_pct"] / 100.0) if p["partial_pct"] > 0 \
            else entry + p["partial_r"] * R
    elif model == "x2" and p["after_hold"] == "trail50":
        trail = panel.col("sma50")

    last = panel.last_row[i_entry]
    hold_until = -1            # x2: set once the fast-gain hold starts; target is off from then on
    pending_open_exit = None   # reason, when a close-based rule fired yesterday
    raise_stop_to = None       # x2 breakeven: new stop level, applied from the next bar
    exit_px, reason, i = None, None, i_entry

    def fill_down(px):
        return px * (1 - slip) - comm

    # Entry bar. After an open fill, a low at the stop came after the fill: stopped.
    # After an intraday buy-stop fill the low may have come first; see entry_day_stop.
    entry_day_hit = not at_close and l[i_entry] <= stop
    if entry_day_hit and entry_raw > o[i_entry] and p["entry_day_stop"] == "path":
        entry_day_hit = c[i_entry] < o[i_entry] or c[i_entry] <= stop
    if entry_day_hit:
        exit_px, reason = fill_down(stop), "stopped (entry day)"
    else:
        i = i_entry
        while True:
            if i >= last:
                reason = "delisted" if panel.delisted[i] else "end of data"
                exit_px = fill_down(c[i])
                break
            i += 1
            bars = i - i_entry
            if raise_stop_to is not None:
                stop, raise_stop_to = max(stop, raise_stop_to), None

            if pending_open_exit is not None:
                exit_px, reason = fill_down(o[i]), pending_open_exit
                break

            # x2: up fast_gain % within fast_bars -> hold at least hold_bars (stop stays)
            if model == "x2" and hold_until < 0 and bars <= p["fast_bars"] and \
                    h[i] >= entry_raw * (1 + p["fast_gain_pct"] / 100.0):
                hold_until = i_entry + int(p["hold_bars"])
                if p["hold_stop"] == "breakeven":
                    raise_stop_to = entry      # from the next bar: today's order of events is unknown
            target_live = model in ("x1", "x2") and hold_until < 0

            # Gaps through the stop or target
            if o[i] <= stop:
                exit_px, reason = fill_down(o[i]), "stopped (gap)"
                break
            if target_live and o[i] >= target:
                exit_px, reason = fill_down(o[i]), "target (gap)"
                break
            if model == "x4" and partial_px is None and o[i] >= part_at:
                partial_px, raise_stop_to = fill_down(o[i]), entry
            # Intraday: stop first (conservative)
            if l[i] <= stop:
                exit_px = fill_down(stop)
                reason = "stopped (breakeven)" if stop >= entry else "stopped"
                break
            if target_live and h[i] >= target:
                exit_px, reason = fill_down(target), "target"
                break
            if model == "x4" and partial_px is None and h[i] >= part_at:
                partial_px, raise_stop_to = fill_down(part_at), entry   # breakeven from the next bar

            # Close-based rules (exit at the next open unless noted)
            if model == "time" and bars >= p["time_bars"]:
                exit_px, reason = fill_down(c[i]), "time exit"
                break
            if model == "x3" and np.isfinite(trail[i]) and c[i] < trail[i]:
                pending_open_exit = f"close below {p['trail']}"
            if model == "x4" and partial_px is not None and np.isfinite(trail[i]) and c[i] < trail[i]:
                pending_open_exit = f"close below {p['trail']}"
            if model == "x2" and hold_until >= 0 and i >= hold_until:
                if p["after_hold"] == "exit":
                    exit_px, reason = fill_down(c[i]), "8-week hold done"
                    break
                if np.isfinite(trail[i]) and c[i] < trail[i]:
                    pending_open_exit = "close below sma50 after hold"
            if p["ts_bars"] and bars == p["ts_bars"] and c[i] < entry * (1 + p["ts_min_gain"] / 100.0):
                pending_open_exit = "time stop"
            if bars >= p["max_hold"]:
                exit_px, reason = fill_down(c[i]), "max hold"
                break

    if partial_px is not None:                  # x4: blend the two sales
        f = p["partial_frac"]
        exit_px = f * partial_px + (1 - f) * exit_px
        reason = f"partial, then {reason}"

    return {
        "entry_row": i_entry, "exit_row": i, "entry": entry, "stop": init_stop,
        "exit": exit_px, "R_pct": R / entry * 100, "result_R": (exit_px - entry) / R,
        "ret_pct": (exit_px / entry - 1) * 100, "exit_reason": reason,
        "bars_held": i - i_entry,
    }


# =============================================================================
# MODE B: every signal taken (one position per symbol at a time)
# =============================================================================

def run_signals(panel, cand, exit_p, signal_start=None, signal_end=None, family="f1"):
    order = np.lexsort((cand["entry_row"], panel.sym[cand["entry_row"]]))
    busy_until = {}
    trades = []
    start = np.datetime64(pd.Timestamp(signal_start)) if signal_start else None
    end = np.datetime64(pd.Timestamp(signal_end)) if signal_end else None
    for k in order:
        ie = cand["entry_row"][k]
        d = panel.date[ie]
        if (start is not None and d < start) or (end is not None and d > end):
            continue
        s = panel.sym[ie]
        if ie <= busy_until.get(s, -1):
            continue
        tr = simulate_trade(panel, ie, cand["entry_raw"][k], cand["at_close"][k],
                            cand["struct_stop"][k], exit_p)
        if tr is None:
            continue
        tr["signal_row"] = cand["signal_row"][k]
        tr["at_close"] = bool(cand["at_close"][k])
        busy_until[s] = tr["exit_row"]
        trades.append(tr)
    return to_frame(panel, trades, family)


def to_frame(panel, trades, family):
    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame(trades)
    t.insert(0, "symbol", panel.sym_names[panel.sym[t["entry_row"]]])
    t.insert(1, "signal_date", panel.date[t["signal_row"]])
    t.insert(2, "entry_date", panel.date[t["entry_row"]])
    t.insert(3, "exit_date", panel.date[t["exit_row"]])
    t["family"] = family
    return t


# =============================================================================
# F0: date-matched random baseline
# =============================================================================

def run_f0(panel, base_mask, family_trades, exit_p, seed):
    """For every family trade, pick a random stock that passed the same
    universe + template filter on the same signal day and buy it on the same
    entry day at the same time of day (the close, or the open), with the same
    risk % as the family trade. Same exits."""
    if family_trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pool_rows = np.flatnonzero(base_mask & panel.has_next)       # sorted by symbol, then date
    pool_rows = pool_rows[np.argsort(panel.date[pool_rows], kind="stable")]
    pool_dates = panel.date[pool_rows]

    ft = family_trades.sort_values("signal_date")
    sig = ft["signal_date"].to_numpy().astype(pool_dates.dtype)
    lo = np.searchsorted(pool_dates, sig, side="left")
    hi = np.searchsorted(pool_dates, sig, side="right")
    risk = ft["R_pct"].to_numpy() / 100.0
    offset = (ft["entry_row"] - ft["signal_row"]).to_numpy()     # 0, 1 or 2 bars after the signal
    at_close = ft["at_close"].to_numpy() if "at_close" in ft else np.zeros(len(ft), dtype=bool)
    n_rows = len(panel.sym)

    busy_until = {}
    trades = []
    for j in range(len(ft)):
        n = hi[j] - lo[j]
        if n <= 0:
            continue
        for _try in range(5):
            r = pool_rows[lo[j] + rng.integers(n)]
            ie = r + offset[j]
            if ie < n_rows and panel.sym[ie] == panel.sym[r] and ie > busy_until.get(panel.sym[r], -1):
                break
        else:
            continue
        raw = panel.c[ie] if at_close[j] else panel.o[ie]
        tr = simulate_trade(panel, ie, raw, bool(at_close[j]), np.nan, exit_p, risk_pct_override=risk[j])
        if tr is None:
            continue
        tr["signal_row"] = r
        tr["at_close"] = bool(at_close[j])
        busy_until[panel.sym[r]] = tr["exit_row"]
        trades.append(tr)
    return to_frame(panel, trades, "f0")


# =============================================================================
# MODE B METRICS
# =============================================================================

def signal_metrics(t):
    if t is None or t.empty:
        return {"trades": 0, "win_rate": 0, "avg_R": 0, "median_R": 0, "total_R": 0,
                "max_dd": 0, "payoff": 0}
    r = t["result_R"]
    wins, losses = r[r > 0], r[r <= 0]
    running = t.sort_values("exit_date")["result_R"].cumsum()
    years = max((t["entry_date"].max() - t["entry_date"].min()).days / 365.25, 1 / 12)
    top = r.sort_values(ascending=False)
    top_n = max(1, int(len(r) * 0.10))
    total = r.sum()
    return {
        "trades":        int(len(t)),
        "win_rate":      round(float((r > 0).mean() * 100), 1),
        "avg_R":         round(float(r.mean()), 3),
        "median_R":      round(float(r.median()), 3),
        "std_R":         round(float(r.std(ddof=0)), 3),
        "payoff":        round(float(wins.mean() / abs(losses.mean())), 3)
                         if len(wins) and len(losses) and losses.mean() != 0 else None,
        "total_R":       round(float(total), 2),
        "max_dd":        round(float((running - running.cummax()).min()), 2),
        "avg_hold":      round(float(t["bars_held"].mean()), 1),
        "trades_per_yr": round(float(len(t) / years), 1),
        "top10_share":   round(float(top.iloc[:top_n].sum() / total), 3) if total > 0 else None,
        "avg_ret_pct":   round(float(t["ret_pct"].mean()), 2),
    }


def yearly_table(t):
    if t.empty:
        return pd.DataFrame()
    y = t.assign(year=pd.to_datetime(t["entry_date"]).dt.year)
    return y.groupby("year")["result_R"].agg(trades="count", avg_R="mean", total_R="sum",
                                             win_rate=lambda x: (x > 0).mean() * 100).round(3)


# =============================================================================
# MODE C: portfolio
# =============================================================================

PORTFOLIO_DEFAULTS = {
    "capital":       100_000.0,
    "max_positions": 8,
    "risk_pct":      0.75,     # % of equity at risk per trade (entry to stop)
    "max_pos_pct":   25.0,     # cap per position, % of equity
    "rank_by":       "rs",     # rs / tight / random — choosing between same-day signals
}


def run_portfolio(panel, trades, params, market=None, seed=0):
    """Accept or reject the Mode B trades as a real account would. Trades are
    independent of the account (exits don't depend on size), so the portfolio
    only decides which ones it can afford and how big they are."""
    p = {**PORTFOLIO_DEFAULTS, **{k: v for k, v in params.items() if k in PORTFOLIO_DEFAULTS}}
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    t = trades.copy()
    if p["rank_by"] == "rs" and "rs_rank" in panel.df:
        t["rank_key"] = -panel.col("rs_rank")[t["signal_row"]]
    elif p["rank_by"] == "tight" and "tight_10" in panel.df:
        t["rank_key"] = panel.col("tight_10")[t["signal_row"]]
    else:
        t["rank_key"] = rng.random(len(t))

    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    dates = pd.DatetimeIndex(np.unique(panel.date))
    dates = dates[dates >= t["entry_date"].min()]
    entries = {pd.Timestamp(d): g.sort_values("rank_key") for d, g in t.groupby("entry_date")}

    # Closing prices for marking open positions to market
    traded = set(t["symbol"])
    sub = panel.df[panel.df["symbol"].isin(traded)]
    closes = sub.pivot(index="date", columns="symbol", values="close").ffill()

    cash, positions, rows, taken = p["capital"], {}, [], []
    for d in dates:
        # Exits first (their cash is available to today's entries)
        cash += close_positions(positions, d)
        # Equity before entries, marked at yesterday's close
        equity = cash + sum(pos["shares"] * pos["last"] for pos in positions.values())
        if d in entries:
            for _, tr in entries[d].iterrows():
                if len(positions) >= p["max_positions"]:
                    break
                risk_per_share = tr["entry"] - tr["stop"] if np.isfinite(tr["stop"]) \
                    else tr["entry"] * NOMINAL_R_PCT / 100
                shares = min(equity * p["risk_pct"] / 100 / risk_per_share,
                             equity * p["max_pos_pct"] / 100 / tr["entry"],
                             cash / tr["entry"])
                shares = int(shares)
                if shares <= 0:
                    continue
                cash -= shares * tr["entry"]
                key = (tr["symbol"], tr["entry_row"])
                positions[key] = {"shares": shares, "exit": tr["exit"], "exit_date": tr["exit_date"],
                                  "symbol": tr["symbol"], "last": tr["entry"]}
                taken.append({**tr.to_dict(), "shares": shares})
        # Trades stopped out on their entry day close the same day
        cash += close_positions(positions, d)
        # Mark to market at today's close
        invested = 0.0
        if d in closes.index:
            row = closes.loc[d]
            for pos in positions.values():
                px = row.get(pos["symbol"])
                if px is not None and np.isfinite(px):
                    pos["last"] = px
                invested += pos["shares"] * pos["last"]
        rows.append({"date": d, "equity": cash + invested, "invested": invested,
                     "positions": len(positions)})
    curve = pd.DataFrame(rows)
    return curve, pd.DataFrame(taken)


def close_positions(positions, d):
    cash = 0.0
    for key in [k for k, pos in positions.items() if pos["exit_date"] <= d]:
        pos = positions.pop(key)
        cash += pos["shares"] * pos["exit"]
    return cash


def portfolio_metrics(curve, market=None):
    if curve.empty:
        return {"cagr": 0, "max_dd_pct": 0, "mar": 0}
    eq = curve["equity"].to_numpy()
    days = (curve["date"].iloc[-1] - curve["date"].iloc[0]).days
    years = max(days / 365.25, 1 / 12)
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = dd.min()
    rets = np.diff(eq) / eq[:-1]
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    # longest drawdown in calendar days
    longest, start = 0, None
    for d, x in zip(curve["date"], dd):
        if x < 0 and start is None:
            start = d
        elif x >= 0 and start is not None:
            longest = max(longest, (d - start).days)
            start = None
    if start is not None:
        longest = max(longest, (curve["date"].iloc[-1] - start).days)
    out = {
        "cagr":          round(cagr * 100, 2),
        "max_dd_pct":    round(max_dd * 100, 2),
        "mar":           round(cagr / abs(max_dd), 3) if max_dd < 0 else None,
        "sharpe":        round(float(sharpe), 3),
        "pct_invested":  round(float((curve["invested"] / curve["equity"]).mean() * 100), 1),
        "longest_dd_days": int(longest),
        "final_equity":  round(float(eq[-1]), 0),
    }
    if market is not None and "spx_close" in market:
        m = market.set_index("date")["spx_close"].reindex(curve["date"]).ffill().dropna()
        if len(m) > 1:
            out["spx_cagr"] = round(float((m.iloc[-1] / m.iloc[0]) ** (1 / years) - 1) * 100, 2)
            out["spx_max_dd_pct"] = round(float((m / m.cummax() - 1).min() * 100), 2)
    return out


def monte_carlo_dd(trade_R, risk_pct=0.75, n=1000, seed=0):
    """Reshuffle trade order to get a range of max drawdowns (in % of equity),
    compounding risk_pct per trade. Returns the 5th / 50th / 95th percentiles."""
    rng = np.random.default_rng(seed)
    r = np.asarray(trade_R) * risk_pct / 100
    dds = []
    for _ in range(n):
        eq = np.cumprod(1 + rng.permutation(r))
        dds.append((eq / np.maximum.accumulate(eq) - 1).min())
    q = np.percentile(dds, [5, 50, 95]) * 100
    return {"mc_dd_p5": round(q[0], 2), "mc_dd_p50": round(q[1], 2), "mc_dd_p95": round(q[2], 2)}
