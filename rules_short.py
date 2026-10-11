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
                    resistance, S5 quiet breakdown (the F3 mirror). S0, the random
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
    "s3_template": "off",     # on = also require the downtrend template (few former
                              # leaders pass it on the day they break the 50-day)
    # S4 rally into resistance
    "s4_line":    "sma50",    # sma50 / ema21, must be falling
    "s4_x":       2.0,        # the day's high comes within X % of the line
    # S5 quiet breakdown (F3 mirror)
    "r":          0.65,       # ATR10 / ATR50
    "v":          0.8,        # vol10 / vol50
    # Shared
    "entry_at":   "close",    # close = short at the close of the first day that closes
                              # below the trigger; next_open = the open after that close;
                              # stop = sell stop at the trigger during the day
    "max_chase":  5.0,        # skip if that close is more than this % below the trigger
}

FAMILIES = ["s1", "s2", "s3", "s4", "s5"]


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
    raise ValueError(f"Unknown short setup {fam!r} (s1-s5)")


def base_params(p):
    """Template settings a setup actually uses. S3 is a former leader breaking its
    50-day, which rarely passes the full downtrend template yet, so S3 (and its
    S0) use the universe + former-leader pool unless s3_template=on."""
    p = {**with_defaults(p, TEMPLATE_DEFAULTS), **{k: v for k, v in p.items() if k in ENTRY_DEFAULTS}}
    if p.get("family") == "s3":
        p["former_leader"] = "on"
        if p.get("s3_template", "off") == "off":
            p["template"] = "off"
    return p


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
        else:
            raise ValueError(f"Unknown short setup {fam!r} (s1-s5)")
    return {"setup": setup, "pivot": pivot, "stop": stop}


def _same_prev(df):
    sym = df["symbol"].astype("category").cat.codes.to_numpy()
    return np.r_[False, sym[1:] == sym[:-1]]
