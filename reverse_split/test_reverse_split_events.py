"""Reverse splits: which days count, the universe on day 0, and the short's sums."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reverse_split_events as rs  # noqa: E402


def toy(symbol, n=300, price=4.0):
    """A flat stock: adjusted and printed close both at `price`, in the universe."""
    c = np.full(n, price)
    return pd.DataFrame({"symbol": symbol, "date": pd.bdate_range("2012-01-02", periods=n), "open": c.copy(),
                         "high": c.copy(), "low": c.copy(), "close": c.copy(), "unadj_close": c.copy(),
                         "volume": 1e6, "turnover": 4e6, "listed": 1, "dv_pct": 90.0, "tight_10": 0.1,
                         "bars_since_break": np.nan, "bad_bar": 0, "bad_bar_dn": 0, "dv50": 4e6, "sma200": price,
                         "in_sp500": 0, "in_sp400": 0, "in_sp600": 0, "in_r2000": 0})


def reverse_split(df, row, ratio):
    """A 1-for-`ratio` reverse split effective at `row`: the printed price before it was
    1/ratio of the adjusted one (the adjusted series stays smooth)."""
    sym = df.loc[row, "symbol"]
    before = (df.index < row) & (df["symbol"] == sym)
    df.loc[before, "unadj_close"] /= ratio


def panel(*frames):
    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    return df, df["symbol"].cat.codes.to_numpy()


def start_of(df, sym):
    return int(df.index[df["symbol"] == sym][0])


def market(df):
    d = pd.DatetimeIndex(np.sort(df["date"].unique()))
    return pd.DataFrame({k: 100.0 for k in ["spx_open", "spx_close", "comp_open", "comp_close"]}, index=d)


P = dict(rs.rules.UNIVERSE_DEFAULTS, min_price=1.0, liq_top_pct=80.0)
ARGS = argparse.Namespace(min_ratio=rs.MIN_RATIO, dedupe=rs.DEDUPE, start=None, end=None)


def events(df, codes):
    ev, _ = rs.select(df, codes, P, ARGS)
    return ev.sort_values("row").reset_index(drop=True)


def test_a_1_for_10_is_found_with_ratio_10_and_a_forward_split_is_not():
    a, b = toy("REV"), toy("FWD")
    df, codes = panel(a, b)
    reverse_split(df, start_of(df, "REV") + 80, 10)
    fwd = (df["symbol"] == "FWD") & (df.index < start_of(df, "FWD") + 80)
    df.loc[fwd, "unadj_close"] *= 2                                 # a 2-for-1 forward split
    ev, counts = rs.find_splits(df, codes)
    assert list(ev["row"]) == [start_of(df, "REV") + 80]
    assert np.isclose(ev["ratio"].iloc[0], 10.0) and not ev["prior_split"].iloc[0]
    assert counts["upward factor jumps on listed bars"] == 1


def test_repeats_short_history_and_bad_prints_are_left_out():
    df, codes = panel(toy("TWO"), toy("LTR"), toy("NEW"), toy("BAD"))
    reverse_split(df, start_of(df, "TWO") + 80, 10)
    reverse_split(df, start_of(df, "TWO") + 110, 5)                  # 30 bars later: a repeat
    reverse_split(df, start_of(df, "LTR") + 80, 10)
    reverse_split(df, start_of(df, "LTR") + 200, 5)                  # 120 bars later: its own event
    reverse_split(df, start_of(df, "NEW") + 30, 10)                  # under 60 bars of history
    spike = start_of(df, "BAD") + 90
    df.loc[spike, "unadj_close"] *= 10                               # one bad print, undone the next bar
    ev, counts = rs.find_splits(df, codes)
    sym = df["symbol"].astype(str).to_numpy()[ev["row"]]
    got = sorted((s, r - start_of(df, s)) for s, r in zip(sym, ev["row"]))
    assert got == [("LTR", 80), ("LTR", 200), ("TWO", 80)]
    assert counts["repeats within the dedupe bars"] == 1
    assert counts["under 60 bars of history"] == 1
    assert counts["bad prints (undone within 5 bars)"] == 1
    later = ev[ev["row"] == start_of(df, "LTR") + 200]
    assert later["prior_split"].iloc[0] and np.isclose(later["ratio"].iloc[0], 5.0)


def test_a_break_on_day_0_is_allowed_but_an_earlier_one_blocks():
    df, codes = panel(toy("DAY0"), toy("EARLY"), toy("CHEAP"))
    for s in ["DAY0", "EARLY", "CHEAP"]:
        reverse_split(df, start_of(df, s) + 80, 10)
    d0 = start_of(df, "DAY0") + 80
    df.loc[d0, ["bad_bar_dn", "bars_since_break"]] = [1, 0]             # the split day fell 25%+
    e = start_of(df, "EARLY")
    df.loc[e + 60:e + 299, "bars_since_break"] = np.arange(240)         # a break 20 bars before day 0
    df.loc[df["symbol"] == "CHEAP", ["unadj_close"]] /= 5               # prints $0.80 on day 0
    ev = events(df, codes)
    assert list(ev["symbol"]) == ["DAY0"]
    ev = rs.features(ev, df, pd.Series(1, index=pd.DatetimeIndex(df["date"].unique())))
    assert bool(ev["break0"].iloc[0]) and np.isclose(ev["px_prev"].iloc[0], 0.4) and np.isclose(ev["px0"].iloc[0], 4)


def test_the_short_at_each_fee_and_the_break_even_fee():
    df, codes = panel(toy("FALL"))
    r = 80
    reverse_split(df, r, 10)
    df.loc[r + 1:, ["open", "high", "low", "close"]] = 3.0               # -25% from the day 0 close
    df.loc[r + 1, ["open", "high"]] = 3.6
    ev = events(df, codes)
    mkt = market(df)
    out = rs.trades(ev, df, mkt, "close0", 20)
    g = pd.concat([ev, pd.DataFrame(out)], axis=1)
    cost = rs.ce.round_trip_cost()
    assert np.isclose(g["g_close0:20"].iloc[0], 25.0 - cost)
    for fee in [5, 25, 50, 100, 200]:
        assert np.isclose(rs.ret(g, "close0:20", fee).iloc[0], 25.0 - cost - fee * 20 / 252)
    assert np.isclose(rs.ret(g, "close0:20", 50, "h").iloc[0], 25.0 - cost - 50 * 20 / 252)   # flat S&P
    assert np.isclose(rs.breakeven(g, "close0:20"), (25.0 - cost) / (20 / 252))
    assert np.isclose(rs.ret(g, "close0:20", rs.breakeven(g, "close0:20")).iloc[0], 0.0)
    op = rs.trades(ev, df, mkt, "open1", 20)
    assert np.isclose(op["g_open1:20"][0], -(3.0 / 3.6 - 1) * 100 - cost)
    assert op["exit_open1:20"][0] == out["exit_close0:20"][0] == r + 20
    late = rs.trades(ev, df, mkt, "close5", 20)
    assert late["exit_close5:20"][0] == r + 25 and np.isclose(late["g_close5:20"][0], -cost)


def test_a_delisted_stock_is_covered_at_its_last_close_and_one_past_the_data_end_is_left_out():
    gone, live = toy("GONE", n=100), toy("LIVE", n=300)
    df, codes = panel(gone, live)
    g0, l0 = start_of(df, "GONE"), start_of(df, "LIVE")
    reverse_split(df, g0 + 80, 10)
    reverse_split(df, l0 + 280, 10)
    df.loc[g0 + 81:g0 + 99, ["open", "high", "low", "close"]] = 2.0     # then stops trading at row 99
    ev = events(df, codes)
    assert list(ev["gone"]) == [True, False]
    out = rs.trades(ev, df, market(df), "close0", 60)
    cost = rs.ce.round_trip_cost()
    assert np.isclose(out["g_close0:60"][0], 50.0 - cost)              # 4.0 -> 2.0, covered at its last close
    assert np.isclose(out["y_close0:60"][0], 19 / 252) and out["del_close0:60"][0] == 1.0
    assert out["exit_close0:60"][0] == g0 + 99
    assert np.isnan(out["g_close0:60"][1])                             # still trading when the data ends


def test_a_downward_break_in_the_hold_is_left_out_an_upward_one_counts_and_squeezes_are_counted():
    df, codes = panel(toy("DN"), toy("UP"), toy("SQZ"))
    for s in ["DN", "UP", "SQZ"]:
        reverse_split(df, start_of(df, s) + 80, 10)
    dn, up, sq = (start_of(df, s) + 80 for s in ["DN", "UP", "SQZ"])
    df.loc[dn + 5:dn + 219, ["open", "high", "low", "close"]] = 1.0
    df.loc[dn + 5, "bad_bar_dn"] = 1                                   # an untrusted gain for a short
    df.loc[up + 5:up + 219, ["open", "high", "low", "close"]] = 6.0
    df.loc[up + 5, "bad_bar"] = 1                                      # a real loss
    df.loc[sq + 3, "high"] = 6.2                                       # 55% above the entry, then back
    ev = events(df, codes)
    out = rs.trades(ev, df, market(df), "close0", 20)
    g = dict(zip(ev["symbol"], out["g_close0:20"]))
    adv = dict(zip(ev["symbol"], out["adv_close0:20"]))
    cost = rs.ce.round_trip_cost()
    assert np.isnan(g["DN"])
    assert np.isclose(g["UP"], -50.0 - cost)
    assert np.isclose(g["SQZ"], -cost) and adv["SQZ"] == 1.0 and adv["UP"] == 1.0


def test_profile_counts_the_drift_after_day_0():
    df, codes = panel(toy("DRIFT"))
    r = 80
    reverse_split(df, r, 10)
    df.loc[r + 1:, "close"] = 4.0 * 0.99 ** np.arange(1, len(df) - r)   # -1% a day after day 0
    ev = events(df, codes)
    prof = rs.profile(ev, df, market(df)).set_index("day")
    assert np.isclose(prof.loc[1, "vs_spx"], 1.0) and np.isclose(prof.loc[1, "running"], 1.0)
    assert np.isclose(prof.loc[10, "running"], 10.0)
    assert prof.loc[-1, "under1_%"] == 100.0 and prof.loc[0, "under1_%"] == 0.0


def test_parse_pick():
    assert rs.parse_pick("close0:60") == ("close0", 60)
    assert rs.parse_pick("open1:5") == ("open1", 5)
    for bad in ["close1:5", "close0:0", "close0"]:
        try:
            rs.parse_pick(bad)
            assert False, bad
        except ValueError:
            pass
