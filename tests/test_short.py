"""Short side: fills, exits, costs, flipped price breaks, the downtrend template,
the S1-S5 setups, the market filter and dividends from the total-return gap."""

import numpy as np
import pandas as pd
import pytest

import build_features as bf
import engine
import engine_short
import rules_short
from data_sources import dividends_from_total_return
from tests.test_engine import panel_from
from tests.test_families import leg, stock, with_features

BASE = {**engine_short.EXIT_DEFAULTS, "slippage_pct": 0.0, "commission": 0.0, "borrow_pct": 0.0}
FLAT = (100, 101, 99, 100)


def short(panel, i_entry=1, entry_raw=100.0, at_close=True, struct_stop=np.nan, **p):
    return engine_short.simulate_short(panel, i_entry, entry_raw, at_close, struct_stop, {**BASE, **p})


def zeros(n):
    return np.zeros(n)


def pn(bars, div=None, **extra):
    n = len(bars)
    cols = {"div": zeros(n) if div is None else div, "bad_bar": zeros(n), "bad_bar_dn": zeros(n),
            "ema21": np.full(n, 1e9), **extra}
    return panel_from(bars, extra=cols)


# ── Fills and exits ──────────────────────────────────────────────────────────

def test_stop_is_above_and_costs_one_R():
    t = short(pn([FLAT, FLAT, (101, 110, 100, 105)]), exit="xs1")
    assert t["stop"] == pytest.approx(107.5)
    assert t["exit_reason"] == "stopped" and t["exit"] == pytest.approx(107.5)
    assert t["result_R"] == pytest.approx(-1.0)


def test_gap_up_through_stop_fills_at_open():
    t = short(pn([FLAT, FLAT, (112, 113, 111, 112)]), exit="xs1")
    assert t["exit_reason"] == "stopped (gap)"
    assert t["result_R"] == pytest.approx(-12 / 7.5)


def test_target_and_gap_target():
    t = short(pn([FLAT, FLAT, (95, 96, 79, 85)]), exit="xs1", fast_bars=0)
    assert t["exit_reason"] == "target" and t["exit"] == pytest.approx(80)
    assert t["result_R"] == pytest.approx(20 / 7.5)
    t = short(pn([FLAT, FLAT, (75, 76, 74, 75)]), exit="xs1", fast_bars=0)
    assert t["exit_reason"] == "target (gap)" and t["exit"] == pytest.approx(75)


def test_stop_and_target_same_bar_is_stop():
    t = short(pn([FLAT, FLAT, (100, 110, 75, 100)]), exit="xs1", fast_bars=0)
    assert t["exit_reason"] == "stopped"


def test_xs2_fast_drop_switches_to_trailing_ema():
    # Down 20% on bar 2 (within 3 weeks): no target; cover at the open after a close above ema21
    bars = [FLAT, FLAT, (95, 96, 79, 82), (82, 83, 78, 80), (80, 86, 79, 85), (84, 85, 83, 84)]
    ema = np.array([1e9, 1e9, 90, 88, 84, 84])
    t = short(pn(bars, ema21=ema))
    assert t["exit_reason"] == "close above ema21"
    assert t["exit"] == pytest.approx(84) and t["bars_held"] == 4


def test_xs2_late_drop_takes_the_target():
    bars = [FLAT, FLAT] + [FLAT] * 16 + [(95, 96, 79, 82)]
    t = short(pn(bars))
    assert t["exit_reason"] == "target" and t["exit"] == pytest.approx(80)


def test_max_hold_covers_at_the_close():
    t = short(pn([FLAT] * 70), max_hold=60)
    assert t["exit_reason"] == "max hold" and t["bars_held"] == 60


def test_entry_at_close_has_no_entry_day_stop():
    t = short(pn([FLAT, (100, 110, 99, 100), FLAT, FLAT]), exit="time", time_bars=2)
    assert t["exit_reason"] == "time exit"


@pytest.mark.parametrize("bar,stopped", [
    ((104, 109, 96, 97), False),    # red bar: the high came before the 100 fill
    ((99, 109, 96, 105), True),     # green bar: down through 100, then up to the stop
])
def test_entry_day_stop_after_intraday_sell_stop(bar, stopped):
    t = short(pn([FLAT, bar, FLAT, FLAT]), at_close=False, exit="time", time_bars=2)
    assert (t["exit_reason"] == "stopped (entry day)") == stopped


def test_delisting_covers_at_the_last_price_as_a_gain():
    bars = [FLAT, FLAT, (60, 61, 40, 45), (44, 45, 1, 2)]
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    df.insert(0, "date", pd.bdate_range("2020-01-01", periods=4))
    df.insert(0, "symbol", "AAA")
    for c in ["div", "bad_bar", "bad_bar_dn"]:
        df[c] = 0.0
    df["ema21"] = 1e9
    p = engine.Panel(df, {"AAA"})
    t = short(p, exit="xs3")
    assert t["exit_reason"] == "delisted" and t["exit"] == pytest.approx(2)
    assert t["result_R"] > 10


# ── Costs ────────────────────────────────────────────────────────────────────

def test_dividends_paid_while_short():
    # Short at the close of bar 1, covered at the close of bar 4: ex-dates 2-4 are owed,
    # not bar 1's (the short came in after it) or bar 5's (already covered)
    div = np.array([0, 2.0, 1.0, 0, 0.5, 3.0])
    t = short(pn([FLAT] * 6, div=div), exit="time", time_bars=3)
    assert t["exit_row"] == 4
    assert t["div_paid"] == pytest.approx(1.5)
    assert t["result_R"] == pytest.approx(-1.5 / 7.5)


def test_dividends_off():
    t = short(pn([FLAT] * 6, div=np.array([0, 0, 1.0, 0, 0, 0])), exit="time", time_bars=3,
              dividends="off")
    assert t["div_paid"] == 0 and t["result_R"] == pytest.approx(0)


def test_huge_dividend_is_a_data_gap_not_a_payout():
    div = np.array([0, 0, 0, 30.0, 0, 0])              # 30% of the price: not a real dividend
    t = short(pn([FLAT, FLAT, (100, 101, 97, 98), (70, 71, 69, 70), FLAT, FLAT], div=div), exit="xs3")
    assert t["exit_reason"] == "dividend gap (data)"
    assert t["exit"] == pytest.approx(98) and t["div_paid"] == 0
    for dividends in ["on", "off"]:                     # no entry on or right after one
        assert short(pn([FLAT] * 5, div=np.array([0, 0, 30.0, 0, 0])), i_entry=2,
                     dividends=dividends) is None
        assert short(pn([FLAT] * 5, div=np.array([0, 0, 30.0, 0, 0])), i_entry=3,
                     dividends=dividends) is None
    t = short(pn([FLAT] * 6, div=np.array([0, 0, 20.0, 0, 0, 0])), exit="time", time_bars=3)
    assert t["div_paid"] == pytest.approx(20)           # 20% is charged as a dividend


def test_borrow_charged_by_calendar_day():
    t = short(pn([FLAT] * 10), exit="time", time_bars=5, borrow_pct=3.65)
    days = 7                                            # 5 business days span a weekend
    assert t["borrow_paid"] == pytest.approx(100 * 0.0365 * days / 365)


def test_slippage_and_commission_go_against_the_short():
    t = short(pn([FLAT] * 4), exit="time", time_bars=2, slippage_pct=0.2, commission=0.01)
    assert t["entry"] == pytest.approx(100 * 0.998 - 0.01)
    assert t["exit"] == pytest.approx(100 * 1.002 + 0.01)


# ── Flipped price breaks ─────────────────────────────────────────────────────

def test_down_break_is_an_untrusted_gain():
    bad_dn = np.array([0, 0, 0, 1, 0])
    t = short(pn([FLAT, FLAT, (100, 101, 97, 98), (5, 6, 4, 5), (5, 6, 4, 5)], bad_bar_dn=bad_dn),
              exit="xs3")
    assert t["exit_reason"] == "price break" and t["exit"] == pytest.approx(98)


def test_up_break_counts_as_a_loss_or_exits():
    bad = np.array([0, 0, 0, 1, 0])
    bars = [FLAT, FLAT, (100, 101, 99, 102), (300, 310, 290, 300), FLAT]
    t = short(pn(bars, bad_bar=bad), exit="xs3")
    assert t["exit_reason"] == "stopped (price break up)" and t["result_R"] < -20
    t = short(pn(bars, bad_bar=bad), exit="xs3", up_break="exit")
    assert t["exit_reason"] == "price break" and t["exit"] == pytest.approx(102)


def test_no_entry_on_or_right_after_a_break():
    for col in ["bad_bar", "bad_bar_dn"]:
        assert short(pn([FLAT] * 5, **{col: np.array([0, 1, 0, 0, 0])})) is None
        assert short(pn([FLAT] * 5, **{col: np.array([0, 0, 0, 0, 0])}), i_entry=2) is not None
        assert short(pn([FLAT] * 5, **{col: np.array([0, 1, 0, 0, 0])}), i_entry=2) is None


def test_open_entry_is_not_skipped_for_an_up_break_on_its_own_bar():
    # the up break shows only at the close, after an open entry: it's a loss, not a skip
    bad = np.array([0, 1, 0, 0])
    bars = [FLAT, (100, 1200, 99, 1100), FLAT, FLAT]
    t = short(pn(bars, bad_bar=bad), at_close=False, exit="time", time_bars=2)
    assert t is not None and t["exit_reason"] == "stopped (entry day)" and t["result_R"] < -0.9
    assert short(pn(bars, bad_bar=bad), exit="time", time_bars=2) is None          # close entry: known
    assert short(pn([FLAT] * 4, bad_bar_dn=bad), at_close=False, exit="time", time_bars=2) is None


def test_stop_too_far_is_skipped():
    assert short(pn([FLAT] * 4), stop_mode="structural", struct_stop=120) is None
    t = short(pn([FLAT] * 4), stop_mode="structural", struct_stop=110, exit="time", time_bars=1)
    assert t["stop"] == pytest.approx(110)


# ── Triggers ─────────────────────────────────────────────────────────────────

def triggers(bars, pivot, entry_at="close", entry=None):
    p = pn(bars)
    n = len(bars)
    setups = {"setup": np.r_[True, np.zeros(n - 1, bool)], "pivot": np.full(n, pivot),
              "stop": np.full(n, 110.0)}
    if entry:
        setups["entry"] = entry
    return engine_short.find_entries(p, setups, {"max_chase": 5.0, "entry_at": entry_at})


def test_short_at_the_close_below_the_trigger():
    c = triggers([FLAT, (99, 99, 96, 97), FLAT], pivot=98)
    assert c["entry_row"].tolist() == [1] and c["entry_raw"][0] == 97
    assert len(triggers([FLAT, (99, 99, 96, 98.5), FLAT], pivot=98)["entry_row"]) == 0   # no close below
    assert len(triggers([FLAT, (99, 99, 90, 92), FLAT], pivot=98)["entry_row"]) == 0     # chased > 5%


def test_sell_stop_and_signal_day_entries():
    c = triggers([FLAT, (99, 99, 96, 99), FLAT], pivot=98, entry_at="stop")
    assert c["entry_raw"][0] == 98
    c = triggers([FLAT, (97, 99, 96, 99), FLAT], pivot=98, entry_at="stop")
    assert c["entry_raw"][0] == 97                       # opened below the trigger
    c = triggers([FLAT, FLAT, FLAT], pivot=np.nan, entry="signal")
    assert c["entry_row"].tolist() == [0] and c["at_close"][0]


# ── Template, market, setups ─────────────────────────────────────────────────

def downtrend_row(**over):
    row = {"close": 50.0, "sma50": 55.0, "sma150": 60.0, "sma200": 65.0, "sma200_dn21": 1,
           "hi52_c": 100.0, "lo52_c": 45.0, "rs_rank": 20.0, "bars_since_leader": 100.0}
    row.update(over)
    return pd.DataFrame([row])


@pytest.mark.parametrize("over,ok", [
    ({}, True),
    ({"close": 61}, False),                  # 1: above the 150-day
    ({"sma150": 70}, False),                 # 2: 150-day above the 200-day
    ({"sma200_dn21": 0}, False),             # 3: 200-day not falling
    ({"sma50": 62}, False),                  # 4: 50-day above the 150-day
    ({"close": 56, "sma50": 55.5}, False),   # 5: above the 50-day
    ({"hi52_c": 60}, False),                 # 6: only 17% below the high
    ({"lo52_c": 35}, False),                 # 7: 43% above the low
    ({"rs_rank": 31}, False),                # 8: RS too high
])
def test_downtrend_template_rules(over, ok):
    assert rules_short.template_mask(downtrend_row(**over), {})[0] == ok


def test_former_leader_rule():
    assert rules_short.template_mask(downtrend_row(), {"former_leader": "on"})[0]
    assert not rules_short.template_mask(downtrend_row(bars_since_leader=300), {"former_leader": "on"})[0]
    assert not rules_short.template_mask(downtrend_row(bars_since_leader=np.nan), {"former_leader": "on"})[0]


def test_market_filter_mirror():
    mk = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3),
                       "spx_above200": [1, 0, 0], "spx_above50": [1, 0, 1], "spx_sma50_rising": [1, 0, 0]})
    d = mk["date"]
    assert rules_short.market_mask(d, mk, {"market": "below200"}).tolist() == [False, True, True]
    assert rules_short.market_mask(d, mk, {"market": "downtrend"}).tolist() == [False, True, False]
    assert rules_short.market_mask(d, mk, {"market": "off"}).tolist() == [True, True, True]


def test_pools():
    tp = rules_short.base_params({"family": "s3"})
    assert tp["template"] == "off" and tp["former_leader"] == "on"
    assert rules_short.base_params({"family": "s3", "pool": "on"})["template"] == "on"
    for fam in ["s6", "s7"]:
        assert rules_short.base_params({"family": fam})["template"] == "off"
    tp = rules_short.base_params({"family": "s1"})
    assert tp["template"] == "on" and tp["former_leader"] == "off"
    assert rules_short.base_params({"family": "s1", "pool": "off"})["template"] == "off"


# A fall from 100 to 70 in 4 weeks on heavy volume, then a light 6-bar bounce to 78
BEAR_PATH = [100.0] * 60 + leg(100, 70, 21) + leg(70, 78, 7)


def bear_volume():
    v = np.full(len(BEAR_PATH), 1e6)
    v[60:80] = 2e6
    v[80:] = 5e5
    return v


def test_bear_flag_features_and_setup():
    df = with_features(stock(BEAR_PATH, bear_volume()))
    last = df.iloc[-1]
    assert last["bflag_bot"] == pytest.approx(70 * 0.998)
    assert last["bflag_age"] == 6
    assert last["bflag_volr20"] < 1
    s = rules_short.entry_setups(df, {"family": "s2"}, np.ones(len(df), bool))
    assert s["setup"][-1]
    assert s["pivot"][-1] == pytest.approx(70 * 0.998)
    assert s["stop"][-1] == pytest.approx(78 * 1.002)


@pytest.mark.parametrize("knobs", [{"flag_d_pct": 35}, {"flag_r": 20}, {"flag_bars": 8}])
def test_bear_flag_knobs_reject(knobs):
    df = with_features(stock(BEAR_PATH, bear_volume()))
    s = rules_short.entry_setups(df, {"family": "s2", **knobs}, np.ones(len(df), bool))
    assert not s["setup"][-1]


def test_s1_s5_trigger_and_stop():
    df = with_features(stock(BEAR_PATH, bear_volume()))
    s = rules_short.entry_setups(df, {"family": "s1", "n": 20}, np.ones(len(df), bool))
    assert s["pivot"][-1] == pytest.approx(df["low_20"].iloc[-1])
    assert s["stop"][-1] == pytest.approx(df["pivot_20"].iloc[-1])
    s = rules_short.entry_setups(df, {"family": "s5", "r": 10, "v": 10}, np.ones(len(df), bool))
    assert s["setup"][-1] and s["pivot"][-1] == pytest.approx(df["low_10"].iloc[-1])


def test_s3_failed_leader_break():
    df = pd.DataFrame({"symbol": "AAA", "close": [101, 100, 97, 96, 95],
                       "sma50": [99, 99, 98, 97.5, 97], "sma50_dn5": [1, 1, 1, 1, 0],
                       "vol_ratio": [1, 1, 2, 2, 2], "pivot_10": [105] * 5})
    s = rules_short.entry_setups(df, {"family": "s3", "s3_v": 1.4}, np.ones(5, bool))
    assert s["setup"].tolist() == [False, False, True, False, False]   # the first close below, on volume
    assert s["stop"][2] == 105 and s["entry"] == "signal"
    s = rules_short.entry_setups(df, {"family": "s3", "s3_v": 2.5}, np.ones(5, bool))
    assert not s["setup"].any()


def test_s4_rally_into_a_falling_line():
    df = pd.DataFrame({"symbol": "AAA", "high": [100, 99, 104], "close": [99, 98, 96],
                       "sma50": [105, 104, 103], "sma50_dn5": [1, 1, 1]})
    s = rules_short.entry_setups(df, {"family": "s4", "s4_x": 2}, np.ones(3, bool))
    assert s["setup"].tolist() == [False, False, True]      # high 104 within 2% of 103, closed below, down
    assert s["stop"][2] == 104


# S6: flat, a breakout from 100 to 130, a pullback to 109 (70% of the move), a rally
# back to 129 and a red day that fails just under the peak
IMPULSE_PATH = [100.0] * 60 + leg(100, 130, 6) + leg(130, 109, 5) + leg(109, 129, 4) + [125.0]


def impulse_stock(fail_bar=(129, 129.5, 124.75)):
    st = stock(IMPULSE_PATH)
    st.loc[len(st) - 1, ["open", "high", "low"]] = list(fail_bar)
    return with_features(st)


def test_impulse_retest_failure():
    df = impulse_stock()
    s = rules_short.entry_setups(df, {"family": "s6"}, np.ones(len(df), bool))
    assert np.flatnonzero(s["setup"]).tolist() == [len(df) - 1]
    assert s["entry"] == "signal"
    peak, low = 130 * 1.002, 100 * 0.998
    assert s["stop"][-1] == pytest.approx(peak + 0.1 * (peak - low), rel=1e-4)


@pytest.mark.parametrize("knobs,bar", [
    ({"s6_dr_min": 0.8}, None),              # the pullback took back only 70%
    ({"s6_retest": 0.5}, None),              # the rally stopped 0.8% short of the peak
    ({"s6_atr": 50}, None),                  # impulse too small for its ATR
    ({}, (129, 131, 124.75)),                # pushed through the peak: not a failure
    ({}, (126, 129.5, 124.75)),              # small red body
])
def test_impulse_knobs_reject(knobs, bar):
    df = impulse_stock(bar or (129, 129.5, 124.75))
    s = rules_short.entry_setups(df, {"family": "s6", **knobs}, np.ones(len(df), bool))
    assert not s["setup"][-1]


# S7: a top at 100 (the 52-week high), a drop and a base, a rally back to 97 (3% under it),
# then a quiet fade to 93
FADE_PATH = ([50.0] * 200 + leg(50, 100, 30) + leg(100, 85, 8) + [85.0] * 30 + leg(85, 97, 6)
             + leg(97, 93, 15))


def test_slow_fade():
    df = with_features(stock(FADE_PATH))
    s = rules_short.entry_setups(df, {"family": "s7"}, np.ones(len(df), bool))
    assert s["setup"][-1]
    assert s["pivot"][-1] == pytest.approx(93 * 0.998, rel=1e-4)     # the fade's low: short below it
    assert s["stop"][-1] == pytest.approx(df["pivot_10"].iloc[-1])


@pytest.mark.parametrize("knobs", [{"s7_x": 2}, {"s7_r": 0.5}, {"s7_min": 15}, {"s7_d_max": 3}])
def test_slow_fade_knobs_reject(knobs):
    df = with_features(stock(FADE_PATH))
    s = rules_short.entry_setups(df, {"family": "s7", **knobs}, np.ones(len(df), bool))
    assert not s["setup"][-1]


def test_lag_stays_inside_a_symbol():
    sym = np.array([0, 0, 0, 1, 1])
    x = np.arange(5.0)
    assert np.allclose(rules_short._lag(x, 1, sym), [np.nan, 0, 1, np.nan, 3], equal_nan=True)
    assert np.allclose(rules_short._at_age(x, np.array([0, 1, 2, 1, 2]), sym),
                       [0, 0, 0, np.nan, np.nan], equal_nan=True)


# ── Features and labels ──────────────────────────────────────────────────────

def test_down_break_feature_and_short_labels():
    close = np.r_[np.full(30, 100.0), np.full(30, 5.0)]          # a 95% one-day drop
    p = stock(close)
    f = bf.symbol_features(p)
    assert f["bad_bar_dn"].iloc[30] == 1 and f["bad_bar"].sum() == 0
    lab = bf.symbol_labels(p, set())
    assert np.isnan(lab["fwd20_s"].iloc[20])                     # spans the drop: left out
    assert np.isfinite(lab["fwd20"].iloc[20])                    # the long label keeps it
    up = stock(np.r_[np.full(30, 5.0), np.full(30, 100.0)])
    lab = bf.symbol_labels(up, set())
    assert np.isfinite(lab["fwd20_s"].iloc[20]) and np.isnan(lab["fwd20"].iloc[20])


def test_dividends_from_total_return():
    close = np.array([100.0, 101.0, 99.0, 99.5])
    div = np.array([0, 0, 1.5, 0])
    tr = [100.0]
    for t in range(1, 4):                       # total return t/t-1 = (close + div) / prior close
        tr.append(tr[-1] * (close[t] + div[t]) / close[t - 1])
    got = dividends_from_total_return(close, np.array(tr))
    assert got == pytest.approx(div, abs=1e-9)
