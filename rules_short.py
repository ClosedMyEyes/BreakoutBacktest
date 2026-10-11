"""
Short-side rules, kept apart from the long side (rules.py)
==========================================================
Follows the Short Side Plan. Nothing here changes a long-side result: the long
scripts never import this file, and every short default lives in its own table.

Every function looks only at columns for day t (the signal day). Orders are for
day t+1 (or the close of day t for the signal-day setups S3 and S4).

Contents
  universe_mask     price / liquidity / size filters (stricter than the long side, to
                    stay where borrow is available), price-break and pinned-price blocks
  market_mask       new shorts only while the S&P 500 is weak (below200 / downtrend)
  template_mask     the downtrend template: Minervini's eight rules mirrored, plus the
                    optional former-leader rule
  entry setups      S1 simple breakdown, S2 bear flag, S3 failed leader, S4 rally into
                    resistance, S5 quiet breakdown (the F3 mirror), S6 failed retest of an
                    impulse high (the daily version of Stockbot's impulse short), S7 slow
                    fade from resistance, then the drop. S0, the random
                    baseline, is built in engine_short.py because it is matched to
                    the days another setup fires.
"""

import numpy as np
import pandas as pd

import config
from rules import SIZE_COLUMNS, with_defaults

# =============================================================================
# DEFAULTS
# =============================================================================

UNIVERSE_DEFAULTS = {
    "min_price":   10.0,    # unadjusted close, $ (long side: 5)
    "liq_top_pct": 30.0,    # top X% of stocks by 50-day dollar volume (long side: 50)
    "size":        "all",   # all / sp500 / sp400 / sp600 / sp400_600 / other
    "block_after_break": 252,
    "min_range10": 2.0,     # skip buyout-pinned stocks (10-day range under X%)
    "market":      "below200",  # off / below200 = S&P 500 below its 200-day on the signal day /
                                # downtrend = also below a falling 50-day
}

MARKET_RULES = {                # market feature -> value it must have
    "off":       {},
    "below200":  {"spx_above200": 0},
    "downtrend": {"spx_above200": 0, "spx_above50": 0, "spx_sma50_rising": 0},
}

TEMPLATE_DEFAULTS = {
    "template":       "on",      # on / off (off = universe only)
    "trend_days":     21,        # rule 3: SMA200 lower than N days ago (21 / 63 / 105)
    "hilo":           "close",   # 52-week high/low from closes or intraday highs/lows
    "min_below_high": 25.0,      # rule 6: at least % below the 52-week high
    "max_above_low":  30.0,      # rule 7: within % of the 52-week low
    "max_rs":         30.0,      # rule 8: RS rank at most this
    "rs_col":         "rs_rank",
    "former_leader":  "off",     # on = also passed the long Trend Template within leader_bars
    "leader_bars":    252,       # 12 months
}

ENTRY_DEFAULTS = {
    "family":     "s1",
    # S1 simple breakdown
    "n":          20,         # trigger = N-day low, stop = N-day high (20 / 50)
    # S2 bear flag
    "flag_d_pct": 20.0,       # drop of at least D % ...
    "flag_w":     4,          # ...within W weeks (4 / 8)
    "flag_r":     50.0,       # the bounce takes back no more than R % of the drop
    "flag_bars":  5,          # bounce lasts at least this many bars
    "flag_vol":   True,       # bounce on lighter average volume than the drop
    # S3 failed leader
    "s3_v":       1.4,        # volume on the break of the 50-day >= V x 50-day average
    # S4 rally into resistance
    "s4_line":    "sma50",    # sma50 / ema21, must be falling
    "s4_x":       2.0,        # the day's high comes within X % of the line
    # S5 quiet breakdown (F3 mirror)
    "r":          0.65,       # ATR10 / ATR50
    "v":          0.8,        # vol10 / vol50
    # S6 impulse retest failure: a breakout to a peak, a deep pullback, a rally back
    # up to the peak, and a red day that fails there
    "s6_bars":    15,         # the peak is 3 to this many bars before the failure day
    "s6_atr":     3.0,        # impulse (peak - lowest low of the 20 bars to the peak) >= this x ATR50
    "s6_dr_min":  0.60,       # pullback takes back at least this share of the impulse...
    "s6_dr_max":  0.95,       # ...and at most this
    "s6_retest":  3.0,        # today's or yesterday's high within X % below the peak, never above it
    "s6_body":    0.5,        # failure day: red, body at least this share of its range
    # S7 slow fade: a top within X % under the 52-week high (resistance, not broken),
    # then a quiet drift down; short the close below the fade's low
    "s7_bars":    30,         # the top is s7_min to this many bars back
    "s7_min":     10,
    "s7_x":       5.0,        # top within X % below the prior 52-week high
    "s7_d_min":   3.0,        # close has drifted this many % below the top...
    "s7_d_max":   15.0,       # ...but not more (slow, not a crash)
    "s7_r":       0.8,        # quiet: ATR10 <= r x ATR50
    # Shared
    "pool":       "auto",     # auto = S1, S2, S4, S5 need the downtrend template; S3, S6 and S7
                              # (strong stocks turning down) use the tradable pool without it
                              # (S3 adds the former-leader rule); on / off forces it
    "entry_at":   "close",    # close = short at the close of the first day that closes
                              # below the trigger; next_open = the open after that close;
                              # stop = sell stop at the trigger during the day
    "max_chase":  5.0,        # skip if that close is more than this % below the trigger
}

FAMILIES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
OWN_POOL = {"s3", "s6", "s7"}          # setups that short stocks not (yet) in a downtrend


# =============================================================================
# COLUMNS NEEDED
# =============================================================================

def universe_columns(p):
    p = with_defaults(p, UNIVERSE_DEFAULTS)
    cols = ["unadj_close", "dv_pct", "listed"]
    cols += SIZE_COLUMNS[p["size"]] or []
    if p["block_after_break"] > 0:
        cols += ["bars_since_break"]
    if p["min_range10"] > 0:
        cols += ["tight_10"]
    return cols


def market_columns(p):
    return list(MARKET_RULES[with_defaults(p, UNIVERSE_DEFAULTS)["market"]])


def template_columns(p):
    p = with_defaults(p, TEMPLATE_DEFAULTS)
    cols = []
    if p["template"] != "off":
        sfx = "c" if p["hilo"] == "close" else "i"
        cols += ["close", "sma50", "sma150", "sma200", f"sma200_dn{int(p['trend_days'])}",
                 f"hi52_{sfx}", f"lo52_{sfx}", p["rs_col"]]
    if p["former_leader"] == "on":
        cols += ["bars_since_leader"]
    return cols


def entry_columns(p):
    p = with_defaults(p, ENTRY_DEFAULTS)
    fam = p["family"]
    if fam == "s1":
        n = int(p["n"])
        return [f"low_{n}", f"pivot_{n}"]
    if fam == "s2":
        wb = int(p["flag_w"]) * 5
        return ["bflag_bot", "bflag_age", "bflag_high", f"bpole_high{wb}", f"bflag_volr{wb}"]
    if fam == "s3":
        return ["close", "sma50", "sma50_dn5", "vol_ratio", "pivot_10", "bars_since_leader"]
    if fam == "s4":
        return ["high", "close", p["s4_line"], f"{p['s4_line']}_dn5"]
    if fam == "s5":
        return ["low_10", "pivot_10", "atr_ratio", "vol10_50"]
    if fam == "s6":
        return ["open", "high", "low", "close", "low_20", "pivot_20", "atr50"]
    if fam == "s7":
        return ["high", "low", "close", "hi52_i", "atr_ratio", "pivot_10"]
    raise ValueError(f"Unknown short setup {fam!r} (s1-s7)")


def base_params(p):
    """Template settings a setup actually uses (its S0 draws from the same pool).
    S3, S6 and S7 short stocks that were strong until recently, which rarely pass
    the downtrend template yet, so with pool=auto they skip it; S3 adds the
    former-leader rule."""
    e = with_defaults(p, ENTRY_DEFAULTS)
    out = {**with_defaults(p, TEMPLATE_DEFAULTS), "family": e["family"], "pool": e["pool"]}
    if e["family"] == "s3":
        out["former_leader"] = "on"
    if e["pool"] == "off" or (e["pool"] == "auto" and e["family"] in OWN_POOL):
        out["template"] = "off"
    elif e["pool"] == "on":
        out["template"] = "on"
    return out


# =============================================================================
# FILTERS
# =============================================================================

def universe_mask(df, params):
    p = with_defaults(params, UNIVERSE_DEFAULTS)
    m = ((df["listed"] == 1) & (df["unadj_close"] >= p["min_price"])
         & (df["dv_pct"] >= 100 - p["liq_top_pct"]))
    cols = SIZE_COLUMNS[p["size"]]
    if cols:
        inside = df[cols].max(axis=1) > 0
        m &= ~inside if p["size"] == "other" else inside
    if p["block_after_break"] > 0:
        m &= ~(df["bars_since_break"] < p["block_after_break"])
    if p["min_range10"] > 0:
        m &= ~(df["tight_10"] * 100 < p["min_range10"])
    return m.fillna(False).to_numpy()


def market_mask(dates, market, params):
    """True on days the market is weak enough for new shorts, judged on the
    signal day's close."""
    rule = MARKET_RULES[with_defaults(params, UNIVERSE_DEFAULTS)["market"]]
    ok = np.ones(len(market), dtype=bool)
    for c, want in rule.items():
        ok &= market[c].to_numpy() == want
    day_ok = pd.Series(ok, index=pd.to_datetime(market["date"]).to_numpy())
    return day_ok.reindex(pd.to_datetime(dates).to_numpy()).fillna(False).to_numpy(dtype=bool)


def template_mask(df, params):
    """The downtrend template on day t (Minervini's eight rules mirrored), plus
    the former-leader rule when it is on."""
    p = with_defaults(params, TEMPLATE_DEFAULTS)
    m = pd.Series(True, index=df.index)
    if p["template"] != "off":
        c, s50, s150, s200 = df["close"], df["sma50"], df["sma150"], df["sma200"]
        sfx = "c" if p["hilo"] == "close" else "i"
        hi, lo = df[f"hi52_{sfx}"], df[f"lo52_{sfx}"]
        m = (
            (c < s150) & (c < s200)                                  # 1
            & (s150 < s200)                                          # 2
            & (df[f"sma200_dn{int(p['trend_days'])}"] == 1)          # 3
            & (s50 < s150) & (s50 < s200)                            # 4
            & (c < s50)                                              # 5
            & (c <= hi * (1 - p["min_below_high"] / 100))            # 6
            & (c <= lo * (1 + p["max_above_low"] / 100))             # 7
            & (df[p["rs_col"]] <= p["max_rs"])                       # 8
        )
    if p["former_leader"] == "on":
        m &= df["bars_since_leader"] < p["leader_bars"]
    return m.fillna(False).to_numpy()


# =============================================================================
# SETUPS
# Each returns numpy arrays aligned with df rows: `setup` (bool), `pivot` (the
# trigger level: short on a close below it) and `stop` (the structural stop,
# above). S3 and S4 add entry="signal": the signal day is the trigger, shorted
# at its close (or the next open).
# =============================================================================

def entry_setups(df, params, base_mask):
    """base_mask = universe & market & template (base_params) on day t."""
    p = with_defaults(params, ENTRY_DEFAULTS)
    fam = p["family"]
    f = lambda c: df[c].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if fam == "s1":
            n = int(p["n"])
            pivot, stop = f(f"low_{n}"), f(f"pivot_{n}")
            setup = base_mask & np.isfinite(pivot)
        elif fam == "s2":
            wb = int(p["flag_w"]) * 5
            bot, age, top = f("bflag_bot"), f("bflag_age"), f("bflag_high")
            pole = f(f"bpole_high{wb}")
            drop = 1 - bot / pole
            bounce = (top - bot) / (pole - bot)
            setup = (base_mask & (age >= p["flag_bars"]) & (age <= config.FLAG_MAX_BARS)
                     & (drop >= p["flag_d_pct"] / 100) & (bounce <= p["flag_r"] / 100))
            if p["flag_vol"]:
                setup &= f(f"bflag_volr{wb}") < 1
            pivot, stop = bot, top
        elif fam == "s3":
            c, s50 = f("close"), f("sma50")
            same_prev = _same_prev(df)
            prev_c, prev_s50 = np.r_[np.nan, c[:-1]], np.r_[np.nan, s50[:-1]]
            crossed = same_prev & (c < s50) & (prev_c >= prev_s50)
            setup = (base_mask & crossed & (f("sma50_dn5") == 1)
                     & (f("vol_ratio") >= p["s3_v"]))
            return {"setup": setup, "pivot": np.full(len(df), np.nan), "stop": f("pivot_10"),
                    "entry": "signal"}
        elif fam == "s4":
            line = f(p["s4_line"])
            c, h = f("close"), f("high")
            prev_c = np.where(_same_prev(df), np.r_[np.nan, c[:-1]], np.nan)
            setup = (base_mask & (f(f"{p['s4_line']}_dn5") == 1)
                     & (h >= line * (1 - p["s4_x"] / 100)) & (c < line) & (c < prev_c))
            return {"setup": setup, "pivot": np.full(len(df), np.nan), "stop": h, "entry": "signal"}
        elif fam == "s5":
            pivot, stop = f("low_10"), f("pivot_10")
            setup = (base_mask & (f("atr_ratio") <= p["r"]) & (f("vol10_50") <= p["v"])
                     & np.isfinite(pivot))
        elif fam == "s6":
            return impulse_setups(df, p, base_mask)
        elif fam == "s7":
            return fade_setups(df, p, base_mask)
        else:
            raise ValueError(f"Unknown short setup {fam!r} (s1-s7)")
    return {"setup": setup, "pivot": pivot, "stop": stop}


def impulse_setups(df, p, base_mask):
    """S6, the daily version of Stockbot's impulse short. On day t:
      peak H      the highest high of the last s6_bars bars before t, at least 3
                  bars back, and a breakout (above the 20-day high before it)
      impulse     H - L, L = the lowest low of the 20 bars up to the peak;
                  at least s6_atr x ATR50 at the peak
      pullback    the lowest low between the peak and t takes back s6_dr_min to
                  s6_dr_max of the impulse
      retest      today's or yesterday's high within s6_retest % below H, and
                  today's high not above H
      failure     today is red with a body of at least s6_body of its range
    Short at today's close; structural stop H + 0.1 x impulse (as in Stockbot)."""
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    sym = _codes(df)
    m = int(p["s6_bars"])
    H = np.full(len(df), -np.inf)
    age = np.zeros(len(df), dtype=np.int64)
    for k in range(1, m + 1):                       # most recent highest high, 1..m bars back
        hk = _lag(h, k, sym)
        new_max = hk > H
        H = np.where(new_max, hk, H)
        age = np.where(new_max, k, age)
    H[~np.isfinite(H)] = np.nan
    L = _at_age(df["low_20"].to_numpy(np.float64), age, sym)
    atr = _at_age(df["atr50"].to_numpy(np.float64), age, sym)
    prior_high = _at_age(df["pivot_20"].to_numpy(np.float64), age + 1, sym)
    pull = np.full(len(df), np.inf)                 # lowest low after the peak, before today
    for k in range(1, m):
        lk = _lag(l, k, sym)
        pull = np.where((k < age) & (lk < pull), lk, pull)
    imp = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        retrace = (H - pull) / imp
        rng = h - l
        retest_hi = np.fmax(h, _lag(h, 1, sym))
        setup = (base_mask & (age >= 3) & (H > prior_high) & (imp >= p["s6_atr"] * atr)
                 & (retrace >= p["s6_dr_min"]) & (retrace <= p["s6_dr_max"])
                 & (retest_hi >= H * (1 - p["s6_retest"] / 100)) & (h <= H)
                 & (c < o) & ((o - c) >= p["s6_body"] * rng))
    return {"setup": setup, "pivot": np.full(len(df), np.nan), "stop": H + 0.1 * imp,
            "entry": "signal"}


def fade_setups(df, p, base_mask):
    """S7 slow fade. On the signal day t:
      top T       the highest high of the last s7_bars bars, s7_min or more bars back
      resistance  T is within s7_x % below the 52-week high before it, and not above it
                  (it came up to resistance and shied away)
      fade        the close is s7_d_min to s7_d_max % below T, and the stock is quiet
                  (ATR10 <= s7_r x ATR50)
    Trigger = the lowest low since T: short the first close below it (the cliff).
    Structural stop = the 10-day high."""
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    sym = _codes(df)
    m = int(p["s7_bars"])
    T = h.copy()
    age = np.zeros(len(df), dtype=np.int64)
    for k in range(1, m + 1):
        hk = _lag(h, k, sym)
        new_max = hk > T
        T = np.where(new_max, hk, T)
        age = np.where(new_max, k, age)
    R = _at_age(df["hi52_i"].to_numpy(np.float64), age + 1, sym)
    low_since = l.copy()                            # lowest low after the top, through today
    for k in range(1, m):
        lk = _lag(l, k, sym)
        low_since = np.where((k < age) & (lk < low_since), lk, low_since)
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = (1 - c / T) * 100
        setup = (base_mask & (age >= p["s7_min"]) & (T <= R) & (T >= R * (1 - p["s7_x"] / 100))
                 & (drift >= p["s7_d_min"]) & (drift <= p["s7_d_max"])
                 & (df["atr_ratio"].to_numpy(np.float64) <= p["s7_r"]))
    return {"setup": setup, "pivot": low_since, "stop": df["pivot_10"].to_numpy(np.float64)}


def _codes(df):
    return df["symbol"].astype("category").cat.codes.to_numpy()


def _lag(x, k, sym):
    """x from k bars earlier in the same symbol (NaN across a symbol start)."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = np.where(sym[k:] == sym[:-k], x[:-k], np.nan)
    return out


def _at_age(x, age, sym):
    """x on the bar `age` bars before each row, same symbol only."""
    idx = np.arange(len(x)) - age
    ok = idx >= 0
    out = np.full(len(x), np.nan)
    out[ok] = np.where(sym[idx[ok]] == sym[ok], x[idx[ok]], np.nan)
    return out


def _same_prev(df):
    sym = _codes(df)
    return np.r_[False, sym[1:] == sym[:-1]]
