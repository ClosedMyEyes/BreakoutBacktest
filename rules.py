"""
Rule definitions shared by the backtester and the live scanner
==============================================================
Everything that decides "does this stock qualify" or "where is the pivot and
stop" lives here, so the scanner can never compute it differently from the
backtest (plan ground rule 3.5).

Every function looks only at columns for day t (the signal day). Orders are for
day t+1. Forward-return columns are never read here.

Contents
  universe_mask     price / liquidity / size filters, price-break block
  price_breaks      days where old and new price history don't connect
  template_mask     Minervini's Trend Template (8 rules, with the Mode A knobs)
  entry setups      F1 simple breakout, F2 tightness (VCP v1), F3 volatility
                    contraction (VCP v2). F4-F6 come in Phase 6.
                    F0 (random baseline) is built in engine.py because it is
                    matched to the dates another family fires.
"""

import numpy as np
import pandas as pd

import config

# =============================================================================
# DEFAULTS
# =============================================================================

UNIVERSE_DEFAULTS = {
    "min_price":   5.0,     # unadjusted close, $
    "liq_top_pct": 50.0,    # keep the top X% of stocks by 50-day dollar volume that day
    "size":        "all",   # all / sp500 / sp400 / sp600 / sp400_600
    "block_after_break": 252,  # no new entries for N bars after a price break (0 = off)
}

TEMPLATE_DEFAULTS = {
    "template":        "on",     # on / off (off = universe only, for the dummy strategy)
    "trend_days":      21,       # rule 3: SMA200 above its value N days ago (21 / 63 / 105)
    "trend_mode":      "vs_ago", # vs_ago, or rising = SMA200 up every day for trend_days
    "hilo":            "close",  # 52-week high/low from closes or intraday highs/lows
    "min_above_low":   30.0,     # rule 6: % above 52-week low
    "max_below_high":  25.0,     # rule 7: within % of 52-week high
    "min_rs":          70.0,     # rule 8
    "rs_col":          "rs_rank",  # rs_rank (weighted quarters) / rs_rank_ibd / rs_rank_252 / rs_rank_126
}

SIZE_COLUMNS = {
    "all":       None,
    "sp500":     ["in_sp500"],
    "sp400":     ["in_sp400"],
    "sp600":     ["in_sp600"],
    "sp400_600": ["in_sp400", "in_sp600"],
}


def with_defaults(params, defaults):
    out = dict(defaults)
    out.update({k: v for k, v in params.items() if k in defaults and v is not None})
    return out


# =============================================================================
# COLUMNS NEEDED (so loaders read only what a run uses)
# =============================================================================

def universe_columns(p):
    p = with_defaults(p, UNIVERSE_DEFAULTS)
    cols = ["unadj_close", "dv_pct", "listed"]
    cols += SIZE_COLUMNS[p["size"]] or []
    if p["block_after_break"] > 0:
        cols += ["bars_since_break"]
    return cols


def template_columns(p):
    p = with_defaults(p, TEMPLATE_DEFAULTS)
    if p["template"] == "off":
        return []
    sfx = "c" if p["hilo"] == "close" else "i"
    trend = "sma200_rising_days" if p["trend_mode"] == "rising" else f"sma200_up{int(p['trend_days'])}"
    return ["close", "sma50", "sma150", "sma200", trend, f"hi52_{sfx}", f"lo52_{sfx}", p["rs_col"]]


# =============================================================================
# FILTERS
# =============================================================================

def universe_mask(df, params):
    p = with_defaults(params, UNIVERSE_DEFAULTS)
    m = ((df["listed"] == 1) & (df["unadj_close"] >= p["min_price"])
         & (df["dv_pct"] >= 100 - p["liq_top_pct"]))
    cols = SIZE_COLUMNS[p["size"]]
    if cols:
        m &= df[cols].max(axis=1) > 0
    if p["block_after_break"] > 0:
        m &= ~(df["bars_since_break"] < p["block_after_break"])     # NaN = never broke
    return m.fillna(False).to_numpy()


def price_breaks(sym, close, unadj_close):
    """True on the bar of a price break (see config.BREAK_*). Uses bars t-1 and t only."""
    move = close / close.groupby(sym, observed=True).shift(1)
    factor = unadj_close / close
    fj = factor / factor.groupby(sym, observed=True).shift(1)
    adj_day = (fj > config.BREAK_FACTOR_JUMP) | (fj < 1 / config.BREAK_FACTOR_JUMP)
    big = (move - 1).abs() * 100 > config.BREAK_MOVE_PCT
    extreme = (move >= config.BREAK_EXTREME) | (move <= 1 / config.BREAK_EXTREME)
    return ((adj_day & big) | extreme).fillna(False).to_numpy()


def template_mask(df, params):
    """All 8 Trend Template rules on day t. Returns a boolean numpy array."""
    p = with_defaults(params, TEMPLATE_DEFAULTS)
    if p["template"] == "off":
        return np.ones(len(df), dtype=bool)
    c, s50, s150, s200 = df["close"], df["sma50"], df["sma150"], df["sma200"]
    sfx = "c" if p["hilo"] == "close" else "i"
    hi, lo = df[f"hi52_{sfx}"], df[f"lo52_{sfx}"]
    if p["trend_mode"] == "rising":
        trend_ok = df["sma200_rising_days"] >= int(p["trend_days"])
    else:
        trend_ok = df[f"sma200_up{int(p['trend_days'])}"] == 1

    m = (
        (c > s150) & (c > s200)                                  # 1
        & (s150 > s200)                                          # 2
        & trend_ok                                               # 3
        & (s50 > s150) & (s50 > s200)                            # 4
        & (c > s50)                                              # 5
        & (c >= lo * (1 + p["min_above_low"] / 100))             # 6
        & (c >= hi * (1 - p["max_below_high"] / 100))            # 7
        & (df[p["rs_col"]] >= p["min_rs"])                       # 8
    )
    return m.fillna(False).to_numpy()


# =============================================================================
# ENTRY FAMILIES
# Each returns a dict of numpy arrays aligned with df rows: `setup` (bool, a
# buy-stop is working for day t+1), `pivot` (buy-stop level) and `stop`
# (structural stop level, NaN when the exit model sets its own stop).
# =============================================================================

ENTRY_DEFAULTS = {
    "family":       "f1",
    # F1 simple breakout
    "n":            50,
    # F2 tightness (VCP v1)
    "k":            10,
    "t":            8.0,      # range % of price
    "x":            15.0,     # within % of 52-week high
    # F3 volatility contraction (VCP v2)
    "r":            0.65,     # ATR10 / ATR50
    "v":            0.8,      # vol10 / vol50
    # Shared
    "max_chase":    5.0,      # skip if day t+1 opens more than this % above the pivot
    "vol_mult":     0.0,      # breakout volume >= this x 50-day average (0 = off)
    "vol_timing":   "a",      # a = buy stop, no volume check; b = enter at close of
                              # breakout day if volume confirms; c = next day's open
}


def entry_columns(params):
    p = with_defaults(params, ENTRY_DEFAULTS)
    fam = p["family"]
    if fam == "f1":
        cols = [f"pivot_{int(p['n'])}"]
    elif fam == "f2":
        k = int(p["k"])
        cols = [f"pivot_{k}", f"low_{k}", f"tight_{k}", "close", "hi52_c"]
    elif fam == "f3":
        cols = ["pivot_10", "low_10", "atr_ratio", "vol10_50"]
    else:
        raise ValueError(f"Unknown entry family {fam!r}")
    if p["vol_timing"] != "a":
        cols += ["vol_ratio"]
    return cols


def entry_setups(df, params, base_mask):
    """base_mask = universe & template on day t."""
    p = with_defaults(params, ENTRY_DEFAULTS)
    fam = p["family"]
    nan = np.full(len(df), np.nan)
    if fam == "f1":
        pivot = df[f"pivot_{int(p['n'])}"].to_numpy()
        setup = base_mask & np.isfinite(pivot)
        stop = nan
    elif fam == "f2":
        k = int(p["k"])
        pivot = df[f"pivot_{k}"].to_numpy()
        stop = df[f"low_{k}"].to_numpy()
        near_high = df["close"].to_numpy() >= df["hi52_c"].to_numpy() * (1 - p["x"] / 100)
        tight = df[f"tight_{k}"].to_numpy() <= p["t"] / 100
        setup = base_mask & near_high & tight & np.isfinite(pivot)
    elif fam == "f3":
        pivot = df["pivot_10"].to_numpy()
        stop = df["low_10"].to_numpy()
        setup = (base_mask & (df["atr_ratio"].to_numpy() <= p["r"])
                 & (df["vol10_50"].to_numpy() <= p["v"]) & np.isfinite(pivot))
    else:
        raise ValueError(f"Unknown entry family {fam!r}")
    return {"setup": setup, "pivot": pivot, "stop": stop}
