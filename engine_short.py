"""
Short-side trade engine, kept apart from the long engine (engine.py)
====================================================================
A short is the long trade mirrored, with the costs a short really pays. It
reuses engine.Panel, the Mode B loop and the metrics; the fill model and exits
are its own, so nothing on the long side changes.

Fill model
  Signal timing     computed at day t close; orders work on day t+1
  entry_at=close    (default) short at the close of the first day that closes
                    below the trigger. Skip if that close is more than max_chase %
                    below it. S3 and S4 short the close of the signal day itself.
  entry_at=next_open short the next open after that close
  entry_at=stop     sell stop at the trigger during the day; if it opens below
                    the trigger, fill at the open. Skip if the open is more than
                    max_chase % below it.
  Stop hit          cover at stop + slippage; if it opens above, at open + slippage
  Stop + target     same day: assume the stop was hit
  Target hit        cover at target + slippage; if it opens below, at the open
  Close-based exits (close above the trailing average) cover at the next open
  Delisting         cover at the last traded price (a bankruptcy is a gain; a
                    buyout's jump up already hit the stop as a gap)
  Price breaks      flipped: a jump DOWN that can't be trusted (bad_bar_dn) is the
                    fake gain, so an open short covers at the prior close. A jump
                    up counts as a real loss (up_break=loss) or covers at the prior
                    close too (up_break=exit). No entry on or right after a break.
  Costs             slippage % per side and commission per printed share, as on
                    the long side; borrow_pct % a year on the short's value for the
                    calendar days held; every dividend going ex while the short is
                    open (div column; prices leave ordinary dividends out).

Results are in R, where R = initial stop level - entry fill.

Exit models
  xs2   (standard, plan "XS2") stop + target. If the stock is down fast_gain_pct
        within fast_bars of entry, the target is switched off and the short is
        covered at the next open after a close above `trail` (ema21).
  xs1   stop + target
  xs3   stop, then cover on a close above `trail`
  time  stop (optional), cover at the close after time_bars
  Every model covers at the close of bar max_hold at the latest (60: shorts move
  fast and squeeze hard). stop_mode=structural uses the setup's own stop.
"""

import numpy as np

import engine

EXIT_DEFAULTS = {
    "exit":          "xs2",
    "stop_mode":     "pct",     # pct / structural
    "stop_pct":      7.5,       # % above the entry fill
    "stop_buffer":   0.0,       # extra % above a structural stop
    "max_risk_pct":  15.0,      # skip setups whose stop is further than this above entry
    "target_pct":    20.0,      # cover at this % profit
    "fast_gain_pct": 20.0,      # xs2: down this % ...
    "fast_bars":     15,        # ...within 3 weeks -> target off, trail instead
    "trail":         "ema21",   # ema21 / sma10 / sma50
    "time_bars":     20,
    "max_hold":      60,
    "entry_day_stop": "path",   # sell stop filled intraday, high reaches the stop that day:
                                # path = a red bar went open-high-low-close (the high came
                                # before the fill: not stopped), a green bar open-low-high-
                                # close (stopped); touch = always stopped
    "up_break":      "loss",    # loss = a jump up while short counts as a real loss;
                                # exit = treat it like a down break (cover at the prior close)
    "borrow_pct":    0.5,       # % a year (test 3)
    "dividends":     "on",      # on = the short pays every dividend going ex while open
    "slippage_pct":  engine.EXIT_DEFAULTS["slippage_pct"],
    "commission":    engine.EXIT_DEFAULTS["commission"],
}

MISSING_DATA_HELP = (
    "The short side needs columns added after the last data build ({cols}). "
    "Run python build_data.py, then python build_features.py, and try again.")


def exit_columns(params):
    p = {**EXIT_DEFAULTS, **{k: v for k, v in params.items() if k in EXIT_DEFAULTS}}
    cols = ["bad_bar", "bad_bar_dn"]
    if p["exit"] in ("xs2", "xs3"):
        cols.append(p["trail"])
    if p["dividends"] == "on":
        cols.append("div")
    return cols


# =============================================================================
# TRIGGERS
# =============================================================================

def find_entries(panel, setups, entry_p):
    """Candidate short entries: (signal_row, entry_row, entry_raw, struct_stop, at_close)."""
    setup, pivot, sstop = setups["setup"], setups["pivot"], setups["stop"]
    o, l, c = panel.o, panel.l, panel.c
    n = len(o)
    t = np.flatnonzero(setup & panel.has_next)
    t1 = t + 1
    piv = pivot[t]
    mc = entry_p["max_chase"] / 100.0
    entry_at = entry_p["entry_at"]
    zeros, ones = np.zeros(len(t), dtype=bool), np.ones(len(t), dtype=bool)

    with np.errstate(invalid="ignore"):
        if setups.get("entry") == "signal":            # S3, S4: the signal day is the trigger
            ok = ones
            if entry_at == "close":
                entry_row, raw, at_close = t, c[t], ones
            else:
                entry_row, raw, at_close = t1, o[t1], zeros
        elif entry_at == "stop":
            ok = (l[t1] < piv) & (o[t1] >= piv * (1 - mc))
            entry_row, raw, at_close = t1, np.minimum(o[t1], piv), zeros
        elif entry_at in ("close", "next_open"):
            conf = c[t1] < piv
            if entry_at == "close":
                ok = conf & (c[t1] >= piv * (1 - mc))
                entry_row, raw, at_close = t1, c[t1], ones
            else:
                has2 = np.zeros(len(t), dtype=bool)
                inside = t1 < n
                has2[inside] = panel.has_next[t1[inside]]
                t2 = np.where(has2, t1 + 1, t1)
                ok = conf & has2 & (o[t2] >= piv * (1 - mc))
                entry_row, raw, at_close = t2, o[t2], zeros
        else:
            raise ValueError(f"entry_at must be close, stop or next_open, not {entry_at!r}")
    ok &= np.isfinite(raw)
    return {
        "signal_row": t[ok], "entry_row": entry_row[ok], "entry_raw": raw[ok],
        "struct_stop": sstop[t][ok], "at_close": at_close[ok],
    }


# =============================================================================
# ONE SHORT
# =============================================================================

def simulate_short(panel, i_entry, entry_raw, at_close, struct_stop, p, risk_pct_override=None):
    """Walk forward from the entry bar until the short is covered. Returns a
    dict, or None if the setup is skipped (stop invalid or too wide)."""
    o, h, l, c = panel.o, panel.h, panel.l, panel.c
    slip = p["slippage_pct"] / 100.0
    comm = p["commission"]
    unadj = panel.col("unadj_close") if "unadj_close" in panel.df else None

    def comm_at(k):
        if unadj is None or not unadj[k] > 0:
            return comm
        return comm * c[k] / unadj[k]

    entry = entry_raw * (1 - slip) - comm_at(i_entry)      # proceeds per share
    if risk_pct_override is not None:
        stop = entry * (1 + risk_pct_override)
    elif p["stop_mode"] == "structural":
        if not np.isfinite(struct_stop):
            return None
        stop = struct_stop * (1 + p["stop_buffer"] / 100.0)
    elif p["stop_pct"] > 0:
        stop = entry * (1 + p["stop_pct"] / 100.0)
    else:
        return None
    if stop <= entry or (stop - entry) / entry > p["max_risk_pct"] / 100.0:
        return None
    R = stop - entry

    def is_break(name, k):
        return name in panel.df and panel.col(name)[k] == 1

    prev_same = i_entry > 0 and panel.has_next[i_entry - 1]
    for name in ("bad_bar", "bad_bar_dn"):            # junk print on the entry or trigger bar
        if is_break(name, i_entry) or (prev_same and is_break(name, i_entry - 1)):
            return None

    model = p["exit"]
    if model not in ("xs1", "xs2", "xs3", "time"):
        raise ValueError(f"Short exit must be xs1, xs2, xs3 or time, not {model!r}")
    target = entry_raw * (1 - p["target_pct"] / 100.0) if model in ("xs1", "xs2") else -np.inf
    trail = panel.col(p["trail"]) if model in ("xs2", "xs3") else None
    last = panel.last_row[i_entry]
    fast_on = False             # xs2: down fast_gain % within fast_bars; target off from then on
    pending_open_exit = None
    exit_px, reason, i = None, None, i_entry

    def fill_up(px):            # a buy to cover on bar i
        return px * (1 + slip) + comm_at(i)

    # Entry bar. After an open fill, a high at the stop came after the fill: stopped.
    entry_day_hit = not at_close and h[i_entry] >= stop
    if entry_day_hit and entry_raw < o[i_entry] and p["entry_day_stop"] == "path":
        entry_day_hit = c[i_entry] > o[i_entry] or c[i_entry] >= stop
    if entry_day_hit:
        exit_px, reason = fill_up(stop), "stopped (entry day)"
    else:
        while True:
            if i >= last:
                reason = "delisted" if panel.delisted[i] else "end of data"
                exit_px = fill_up(c[i])
                break
            i += 1
            if is_break("bad_bar_dn", i) or (p["up_break"] == "exit" and is_break("bad_bar", i)):
                i -= 1                                 # a drop we can't trust: out at the last good close
                exit_px, reason = fill_up(c[i]), "price break"
                break
            bars = i - i_entry
            if pending_open_exit is not None:
                exit_px, reason = fill_up(o[i]), pending_open_exit
                break

            if model == "xs2" and not fast_on and bars <= p["fast_bars"] and \
                    l[i] <= entry_raw * (1 - p["fast_gain_pct"] / 100.0):
                fast_on = True
            target_live = model in ("xs1", "xs2") and not fast_on

            up_brk = is_break("bad_bar", i)
            if o[i] >= stop:
                exit_px = fill_up(o[i])
                reason = "stopped (price break up)" if up_brk else "stopped (gap)"
                break
            if target_live and o[i] <= target:
                exit_px, reason = fill_up(o[i]), "target (gap)"
                break
            if h[i] >= stop:                           # stop first (conservative)
                exit_px, reason = fill_up(stop), "stopped"
                break
            if target_live and l[i] <= target:
                exit_px, reason = fill_up(target), "target"
                break

            if model == "time" and bars >= p["time_bars"]:
                exit_px, reason = fill_up(c[i]), "time exit"
                break
            if (model == "xs3" or (model == "xs2" and fast_on)) and np.isfinite(trail[i]) \
                    and c[i] > trail[i]:
                pending_open_exit = f"close above {p['trail']}"
            if bars >= p["max_hold"]:
                exit_px, reason = fill_up(c[i]), "max hold"
                break

    # Costs only a short pays
    div_paid = 0.0
    if p["dividends"] == "on" and i > i_entry:
        div_paid = float(np.nansum(panel.col("div")[i_entry + 1:i + 1]))
    days = (panel.date[i] - panel.date[i_entry]) / np.timedelta64(1, "D")
    borrow = entry_raw * p["borrow_pct"] / 100.0 * days / 365.0
    pnl = entry - exit_px - div_paid - borrow
    return {
        "entry_row": i_entry, "exit_row": i, "entry": entry, "stop": stop,
        "exit": exit_px, "div_paid": div_paid, "borrow_paid": borrow,
        "R_pct": R / entry * 100, "result_R": pnl / R, "ret_pct": pnl / entry * 100,
        "exit_reason": reason, "bars_held": i - i_entry,
    }


# =============================================================================
# MODE B AND S0 (the long loops, with the short simulator)
# =============================================================================

def run_signals(panel, cand, exit_p, signal_start=None, signal_end=None, family="s1"):
    return engine.run_signals(panel, cand, exit_p, signal_start, signal_end, family,
                              sim=simulate_short)


def run_s0(panel, base_mask, family_trades, exit_p, seed):
    """S0: for every setup trade, a random stock from the same pool (universe +
    market + downtrend template) on the same signal day, shorted on the same
    entry day at the same time of day with the same risk %. Same exits."""
    out = engine.run_f0(panel, base_mask, family_trades, exit_p, seed, sim=simulate_short)
    if len(out):
        out["family"] = "s0"
    return out


def check_columns(have, need):
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(MISSING_DATA_HELP.format(cols=", ".join(missing)))


def cost_summary(t):
    """Average dividend and borrow paid per trade, in R."""
    if t.empty:
        return {}
    R = t["R_pct"] / 100 * t["entry"]
    return {"div_R": round(float((t["div_paid"] / R).mean()), 3),
            "borrow_R": round(float((t["borrow_paid"] / R).mean()), 3)}
