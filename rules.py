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
                    contraction (VCP v2), F4 full VCP (swing points), F5 flag,
                    F6 pullback / retest of an earlier breakout.
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
# (structural stop level, NaN when the exit model sets its own stop). F6 adds
# entry="open": buy at the next open instead of a buy-stop.
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
    # F4 full VCP (v3)
    "vcp_z":        3,        # zigzag swing size % (one of config.VCP_SWING_PCTS)
    "vcp_c":        0.65,     # each pullback no more than c x the one before
    "vcp_f":        10.0,     # final pullback no deeper than f %
    "vcp_min":      2,        # contractions needed (2-4)
    "vcp_base":     15,       # base lasts at least this many bars (3 weeks)
    "vcp_vol":      True,     # final pullback on lower average volume than the one before
    # F5 flag
    "flag_a":       50.0,     # pole: advance of at least a %
    "flag_w":       8,        # ...within w weeks (4 / 8)
    "flag_r":       33.0,     # flag pulls back no more than r % of the advance
    "flag_d":       5,        # flag lasts at least d bars (at most config.FLAG_MAX_BARS)
    # F6 pullback / retest (entry at the next open, stop = retest-day low)
    "rt_base":      "f1",     # family whose breakout is retested (f1-f5, with its own knobs)
    "rt_m":         10,       # retest within m bars of the breakout
    "rt_x":         2.0,      # the low comes within x % of the level
    "rt_ref":       "pivot",  # level: pivot / ema21 / sma10
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
    elif fam == "f4":
        z = int(p["vcp_z"])
        cols = [f"vcp{z}_{c}" for c in ["pivot", "low", "d1", "d2", "d3", "d4",
                                         "hpos2", "hpos3", "hpos4", "volr"]] + ["bars"]
    elif fam == "f5":
        cols = ["flag_top", "flag_age", "flag_low", f"pole_low{int(p['flag_w']) * 5}"]
    elif fam == "f6":
        if p["rt_base"] == "f6":
            raise ValueError("rt_base must be f1-f5")
        cols = entry_columns({**p, "family": p["rt_base"], "vol_timing": "a"})
        cols += ["high", "low", "close"] + ([p["rt_ref"]] if p["rt_ref"] != "pivot" else [])
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
    elif fam == "f4":
        return vcp_setups(df, p, base_mask)
    elif fam == "f5":
        return flag_setups(df, p, base_mask)
    elif fam == "f6":
        return retest_setups(df, p, base_mask)
    else:
        raise ValueError(f"Unknown entry family {fam!r}")
    return {"setup": setup, "pivot": pivot, "stop": stop}


def vcp_setups(df, p, base_mask):
    """F4: 2-4 pullbacks, each no more than c x the one before, final one no
    deeper than f %, base of at least vcp_base bars, quieter final pullback.
    Pivot = top of the final contraction, stop = its low."""
    z = int(p["vcp_z"])
    g = lambda name: df[f"vcp{z}_{name}"].to_numpy(np.float64)
    d1, d2, d3, d4 = g("d1"), g("d2"), g("d3"), g("d4")
    c = p["vcp_c"]
    with np.errstate(invalid="ignore"):
        chain2 = d1 <= c * d2
        chain3 = chain2 & (d2 <= c * d3)
        chain4 = chain3 & (d3 <= c * d4)
    n_con = np.where(chain4, 4, np.where(chain3, 3, np.where(chain2, 2, 1)))
    first_high = np.where(n_con == 4, g("hpos4"), np.where(n_con == 3, g("hpos3"), g("hpos2")))
    base_len = df["bars"].to_numpy(np.float64) - first_high
    pivot = g("pivot")
    with np.errstate(invalid="ignore"):
        setup = (base_mask & np.isfinite(pivot) & (n_con >= int(p["vcp_min"]))
                 & (d1 <= p["vcp_f"] / 100) & (base_len >= p["vcp_base"]))
        if p["vcp_vol"]:
            setup &= g("volr") < 1
    return {"setup": setup, "pivot": pivot, "stop": g("low")}


def flag_setups(df, p, base_mask):
    """F5: an advance of at least a % within w weeks to the pole top, then a
    flag of at least d bars that gives back no more than r % of the advance.
    Pivot = flag high (the pole top), stop = flag low."""
    top = df["flag_top"].to_numpy(np.float64)
    age = df["flag_age"].to_numpy(np.float64)
    low = df["flag_low"].to_numpy(np.float64)
    pole = df[f"pole_low{int(p['flag_w']) * 5}"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        adv = top / pole - 1
        depth = (top - low) / (top - pole)
        setup = (base_mask & (age >= p["flag_d"]) & (age <= config.FLAG_MAX_BARS)
                 & (adv >= p["flag_a"] / 100) & (depth <= p["flag_r"] / 100))
    return {"setup": setup, "pivot": top, "stop": low}


def retest_setups(df, p, base_mask):
    """F6: after a breakout from family rt_base (the first one in rt_m bars),
    the first day within rt_m bars whose low comes within rt_x % of the level
    (the breakout pivot, or a moving average) and that closes above the level
    and above the prior close. Entry at the next open, stop = that day's low."""
    bs = entry_setups(df, {**p, "family": p["rt_base"], "vol_timing": "a"}, base_mask)
    n = len(df)
    sym = df["symbol"].astype("category").cat.codes.to_numpy()
    same_prev = np.r_[False, sym[1:] == sym[:-1]]
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    prev_setup = np.r_[False, bs["setup"][:-1]] & same_prev
    prev_piv = np.r_[np.nan, bs["pivot"][:-1]]
    with np.errstate(invalid="ignore"):
        brk = prev_setup & (hi > prev_piv)

    # A breakout opens a retest window unless another one opened in the last m bars
    m = int(p["rt_m"])
    idx = np.arange(n)
    seg_start = np.maximum.accumulate(np.where(~same_prev, idx, 0))     # symbol's first row
    rows = np.flatnonzero(brk)
    start = np.zeros(n, dtype=bool)
    last_open = -10 ** 9
    for r in rows:                                  # breakouts are few; a loop is fine
        if r - last_open > m or last_open < seg_start[r]:
            start[r] = True
            last_open = r
    win = np.maximum.accumulate(np.where(start, idx, -1))
    in_sym = (win >= seg_start) & (win >= 0)
    age = np.where(in_sym, idx - win, -1)
    level = (np.where(in_sym, prev_piv[np.maximum(win, 0)], np.nan) if p["rt_ref"] == "pivot"
             else df[p["rt_ref"]].to_numpy(np.float64))
    prev_cl = np.r_[np.nan, cl[:-1]]
    x = p["rt_x"] / 100
    with np.errstate(invalid="ignore"):
        retest = ((age >= 1) & (age <= m) & (lo <= level * (1 + x)) & (lo >= level * (1 - x))
                  & (cl > level) & same_prev & (cl > prev_cl))
    # first retest in each window only
    count = pd.Series(retest.astype(np.int32)).groupby(np.where(in_sym, win, -1)).cumsum().to_numpy()
    setup = base_mask & retest & (count == 1)
    return {"setup": setup, "pivot": np.full(n, np.nan), "stop": lo, "entry": "open"}
