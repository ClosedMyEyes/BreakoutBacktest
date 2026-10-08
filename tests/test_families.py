"""F4 full VCP, F5 flag, F6 retest and the X4 partial exit on hand-built paths."""

import numpy as np
import pandas as pd
import pytest

import build_features as bf
import engine
import rules
from tests.test_engine import BASE, panel_from


def leg(a, b, n):
    return list(np.linspace(a, b, n))[1:]


def stock(close, vol=None, spread=0.002):
    close = np.asarray(close, float)
    n = len(close)
    return pd.DataFrame({
        "symbol": "AAA", "date": pd.bdate_range("2020-01-01", periods=n),
        "open": close, "high": close * (1 + spread), "low": close * (1 - spread), "close": close,
        "volume": 1e6 if vol is None else vol, "unadj_close": close, "turnover": close * 1e6,
    })


def with_features(p):
    f = bf.symbol_features(p)
    return pd.concat([p, f], axis=1)


# A base after an uptrend: pullbacks of 20%, 10%, then 5% on drying volume
VCP_PATH = ([60.0] + leg(60, 100, 30) + leg(100, 80, 10) + leg(80, 98, 10) + leg(98, 88.2, 8)
            + leg(88.2, 96, 8) + leg(96, 91.2, 6) + leg(91.2, 95, 4))


def vcp_volume():
    v = np.full(len(VCP_PATH), 1e6)
    v[-10:] = 4e5
    return v


def test_vcp_finds_three_contractions():
    df = with_features(stock(VCP_PATH, vcp_volume()))
    s = rules.entry_setups(df, {"family": "f4", "vcp_min": 3}, np.ones(len(df), bool))
    assert s["setup"][-1]
    assert s["pivot"][-1] == pytest.approx(96 * 1.002, rel=1e-3)   # top of the final contraction
    assert s["stop"][-1] < 91.2


@pytest.mark.parametrize("knobs", [
    {"vcp_f": 4},                 # final pullback is about 5%
    {"vcp_min": 4},               # only three contractions
    {"vcp_c": 0.4},               # 10% is not <= 0.4 x 20%
])
def test_vcp_knobs_reject(knobs):
    df = with_features(stock(VCP_PATH, vcp_volume()))
    s = rules.entry_setups(df, {"family": "f4", **knobs}, np.ones(len(df), bool))
    assert not s["setup"][-1]


def test_vcp_needs_quieter_final_pullback():
    df = with_features(stock(VCP_PATH))                       # flat volume
    s = rules.entry_setups(df, {"family": "f4"}, np.ones(len(df), bool))
    assert not s["setup"][-1]
    assert rules.entry_setups(df, {"family": "f4", "vcp_vol": False}, np.ones(len(df), bool))["setup"][-1]


# Flat, then +60% in 6 weeks, then an 8-bar flag that gives back about 25% of it
FLAG_PATH = [50.0] * 60 + leg(50, 80, 31) + leg(80, 72.5, 5) + leg(72.5, 76, 4)


def test_flag_setup_and_levels():
    df = with_features(stock(FLAG_PATH))
    s = rules.entry_setups(df, {"family": "f5", "flag_a": 50, "flag_w": 8, "flag_r": 33, "flag_d": 5},
                           np.ones(len(df), bool))
    assert s["setup"][-1]
    assert s["pivot"][-1] == pytest.approx(80 * 1.002, rel=1e-3)
    assert s["stop"][-1] == pytest.approx(72.5 * 0.998, rel=1e-3)
    assert not s["setup"][-5]                                  # flag only 4 bars old there


@pytest.mark.parametrize("knobs", [{"flag_a": 100}, {"flag_r": 15}, {"flag_d": 12}, {"flag_w": 4}])
def test_flag_knobs_reject(knobs):
    df = with_features(stock(FLAG_PATH))
    p = {"family": "f5", "flag_a": 50, "flag_w": 8, "flag_r": 33, "flag_d": 5, **knobs}
    assert not rules.entry_setups(df, p, np.ones(len(df), bool))["setup"][-1]


def test_retest_after_breakout():
    close = [100.0] * 30 + [106, 108, 110, 104, 104]
    p = stock(close, spread=0.01)                               # 20-day pivot = 101, broken on bar 30
    p.loc[34, ["low", "close"]] = [101.5, 105.0]                # dips to the pivot, closes above it and up
    df = with_features(p)
    s = rules.entry_setups(df, {"family": "f6", "rt_base": "f1", "n": 20, "rt_m": 10, "rt_x": 2},
                           np.ones(len(df), bool))
    assert np.flatnonzero(s["setup"]).tolist() == [34]
    assert s["stop"][34] == pytest.approx(101.5)
    assert s["entry"] == "open"


def test_open_entry_fills_at_next_open():
    pn = panel_from([(100, 101, 99, 100), (103, 104, 102, 103)])
    s = {"setup": np.array([True, False]), "pivot": np.full(2, np.nan),
         "stop": np.array([99.0, np.nan]), "entry": "open"}
    cand = engine.find_entries(pn, s, {"max_chase": 5.0, "entry_at": "stop"})
    assert cand["entry_row"].tolist() == [1] and cand["entry_raw"].tolist() == [103]


def test_x4_partial_then_trail():
    flat = (100, 101, 99, 100)
    bars = [flat, flat, (110, 116, 109, 114), (114, 115, 112, 113), (113, 114, 111, 111), (112, 113, 111, 112)]
    trail = np.full(len(bars), 105.0)
    trail[4] = 112.0                                            # close 111 < 112 -> sell next open (112)
    pn = panel_from(bars, extra={"sma50": trail})
    t = engine.simulate_trade(pn, 1, 100.0, False, np.nan,
                              {**BASE, "exit": "x4", "partial_frac": 0.33, "partial_r": 2.0})
    assert t["exit_reason"] == "partial, then close below sma50"
    assert t["exit"] == pytest.approx(0.33 * 115 + 0.67 * 112)
    assert t["result_R"] == pytest.approx((0.33 * 115 + 0.67 * 112 - 100) / 7.5)


def test_x4_stop_moves_to_breakeven_after_partial():
    flat = (100, 101, 99, 100)
    bars = [flat, flat, (110, 116, 109, 114), (105, 106, 99, 100), flat]
    pn = panel_from(bars, extra={"sma50": np.full(len(bars), 90.0)})
    t = engine.simulate_trade(pn, 1, 100.0, False, np.nan,
                              {**BASE, "exit": "x4", "partial_frac": 0.5, "partial_r": 2.0})
    assert t["exit_reason"] == "partial, then stopped (breakeven)"
    assert t["exit"] == pytest.approx(0.5 * 115 + 0.5 * 100)


def test_retest_bought_at_the_close_by_default():
    pn = panel_from([(100, 101, 99, 100), (103, 104, 102, 103)])
    s = {"setup": np.array([True, False]), "pivot": np.full(2, np.nan),
         "stop": np.array([99.0, np.nan]), "entry": "open"}
    cand = engine.find_entries(pn, s, {"max_chase": 5.0, "entry_at": "close", "vol_mult": 0})
    assert cand["entry_row"].tolist() == [0] and cand["entry_raw"].tolist() == [100]
    assert cand["at_close"].tolist() == [True]


def test_close_entry_needs_no_volume_column_and_respects_chase():
    bars = [(95, 100, 94, 99), (99, 106, 98, 104), (104, 105, 103, 104)]
    pn = panel_from(bars)                                       # no vol_ratio column
    s = {"setup": np.array([True, False, False]), "pivot": np.array([100.0, np.nan, np.nan]),
         "stop": np.full(3, np.nan)}
    cand = engine.find_entries(pn, s, {"max_chase": 5.0, "entry_at": "close", "vol_mult": 0})
    assert cand["entry_row"].tolist() == [1] and cand["entry_raw"].tolist() == [104]
    cand = engine.find_entries(pn, s, {"max_chase": 3.0, "entry_at": "close", "vol_mult": 0})
    assert len(cand["entry_row"]) == 0                          # closed 4% above the pivot


def test_f0_buys_at_the_same_time_of_day_as_the_family():
    flat = (100, 101, 99, 100)
    a = panel_from([flat] * 6, symbol="AAA").df
    b = panel_from([(50, 51, 49, 50), (50, 56, 45, 55), (55, 56, 54, 55), (55, 56, 54, 55),
                    (55, 56, 54, 55), (55, 56, 54, 55)], symbol="BBB").df
    pn = engine.Panel(pd.concat([a, b], ignore_index=True), set())
    fam = pd.DataFrame({"signal_date": [pn.date[0]], "signal_row": [0], "entry_row": [1],
                        "R_pct": [7.5], "at_close": [True]})
    base = np.zeros(len(pn.o), dtype=bool)
    base[6] = True                                              # only BBB passes on the signal day
    f0 = engine.run_f0(pn, base, fam, {**BASE, "exit": "time", "time_bars": 2}, seed=0)
    assert f0["entry_row"].tolist() == [7]
    assert f0["entry"].iloc[0] == pytest.approx(55)            # BBB's close, not its 50 open
    assert f0["exit_reason"].iloc[0] == "time exit"            # its 45 low came before the close buy
