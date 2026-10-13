"""Crash-keeps-falling short as real trades: the event, the two entries, the stop,
and the spike-short overlap marks."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import engine  # noqa: E402
import engine_short  # noqa: E402
import spike_short as ss  # noqa: E402
import crash_trades as ct  # noqa: E402


def panel_from_closes(closes, opens=None, highs=None):
    c = np.asarray(closes, float)
    o = c if opens is None else np.asarray(opens, float)
    h = np.maximum(c, o) if highs is None else np.asarray(highs, float)
    n = len(c)
    z = np.zeros(n)
    df = pd.DataFrame({"symbol": "AAA", "date": pd.bdate_range("2020-01-01", periods=n),
                       "open": o, "high": h, "low": np.minimum(c, o), "close": c,
                       "volume": 1e6, "sma200": np.nan, "in_sp500": z, "in_sp400": z, "in_sp600": z,
                       "bad_bar": z, "bad_bar_dn": z, "div": z, "unadj_close": c, "dv_pct": 50.0})
    return engine.Panel(df, set())


# 70 bars rising 10 -> 14 (a 33% run over the 60 bars before the fall), then 12, 10.5, 9.5:
# the 3-day fall reaches 30% on bar 72 (9.5 / 14), not on bar 71 (10.5 / 13.9 = -25%)
RUN = list(np.linspace(10, 14, 70))
CRASH = RUN + [12, 10.5, 9.5] + [9.0] * 15
OPENS = list(CRASH)
OPENS[73] = 9.2


def test_event_is_the_first_day_the_fall_reaches_30_after_a_run():
    p = panel_from_closes(CRASH, OPENS)
    ev = ct.crash_events(p, np.ones(len(CRASH), dtype=bool))
    assert list(ev["row"]) == [72]
    assert ev["run60"][0] == pytest.approx((14 / RUN[9] - 1) * 100)


def test_no_run_up_no_event():
    flat = [14.0] * 70 + [12, 10.5, 9.5] + [9.0] * 15
    p = panel_from_closes(flat)
    assert ct.crash_events(p, np.ones(len(flat), dtype=bool)).empty


def test_close_and_next_open_entries():
    p = panel_from_closes(CRASH, OPENS)
    ev = ct.crash_events(p, np.ones(len(CRASH), dtype=bool))
    c = ct.crash_candidates(p, ev, "close")
    assert list(c["entry_row"]) == [72] and c["entry_raw"][0] == pytest.approx(9.5) and c["at_close"][0]
    c = ct.crash_candidates(p, ev, "next_open")
    assert list(c["signal_row"]) == [72] and list(c["entry_row"]) == [73]
    assert c["entry_raw"][0] == pytest.approx(9.2) and not c["at_close"][0]


def test_no_next_bar_no_open_entry():
    p = panel_from_closes(CRASH[:73])                    # stops trading on the crash day
    ev = ct.crash_events(p, np.ones(73, dtype=bool))
    assert list(ev["row"]) == [72]
    assert len(ct.crash_candidates(p, ev, "next_open")["entry_row"]) == 0


def test_a_squeeze_hits_the_stop_and_no_squeeze_is_covered_after_10_days():
    highs = list(np.maximum(CRASH, OPENS))
    highs[76] = 12.5                                     # 32% above the 9.5 short: through a 25% stop
    p = panel_from_closes(CRASH, OPENS, highs)
    ev = ct.crash_events(p, np.ones(len(CRASH), dtype=bool))
    cand = ct.crash_candidates(p, ev, "close")
    xp = {**ss.exit_params(25, 10, 0.0), "slippage_pct": 0.0, "commission": 0.0}
    t = engine_short.run_signals(p, cand, xp)
    assert t["exit_reason"][0] == "stopped" and t["ret_pct"][0] == pytest.approx(-25)
    t = engine_short.run_signals(panel_from_closes(CRASH, OPENS), cand, xp)
    assert t["exit_reason"][0] == "time exit" and t["exit_row"][0] == 82
    assert t["ret_pct"][0] == pytest.approx((9.5 - 9.0) / 9.5 * 100)


def test_count_before_stays_inside_one_stock_and_window():
    sym = np.array([0] * 10 + [1] * 10)
    flag = np.zeros(20, dtype=bool)
    flag[[2, 8]] = True                                   # stock 0 only
    assert list(ct.count_before(flag, sym, np.array([9, 9, 12]), 1)) == [0, 0, 0]
    assert list(ct.count_before(flag, sym, np.array([8, 9, 12]), 5)) == [1, 1, 0]
    assert list(ct.count_before(flag, sym, np.array([9]), 20)) == [2]


def test_spike_overlap_marks():
    p = panel_from_closes(CRASH, OPENS)
    ev = ct.crash_events(p, np.ones(len(CRASH), dtype=bool))
    t = engine_short.run_signals(p, ct.crash_candidates(p, ev, "close"),
                                 ss.exit_params(25, 10, 0.0))
    spx = pd.Series(1.0, index=pd.bdate_range("2020-01-01", periods=len(CRASH)))
    flag = np.zeros(len(CRASH), dtype=bool)
    flag[60] = True                                       # a spike setup 12 bars before the crash
    f = ct.entry_features(p, t, ev, flag, spx)
    assert f["spike-short setup on it, past 20 bars"][0] == "yes"
    flag[:] = False
    flag[30] = True                                       # 42 bars before: same year, not the same move
    f = ct.entry_features(p, t, ev, flag, spx)
    assert f["spike-short setup on it, past 20 bars"][0] == "no"
    assert f["spike-short setup on it, past year"][0] == "yes"


def test_spike_setups_flag_the_live_rules_gap_entry():
    closes = [10.0] * 12 + [14, 18, 21] + [21.0] * 10      # bar 14 closes 110% above the 10-bar low
    opens = list(closes)
    opens[15] = 22                                         # bar 15 opens 4.8% above bar 14's high (21)
    p = panel_from_closes(closes, opens)
    flag, cand = ct.spike_setups(p, np.ones(len(closes), dtype=bool))
    assert list(np.flatnonzero(flag)) == [15] and list(cand["signal_row"]) == [14]
    opens[15] = 21.3                                       # only 1.4% above: under the 2% gap
    flag, _ = ct.spike_setups(panel_from_closes(closes, opens), np.ones(len(closes), dtype=bool))
    assert not flag.any()


NO_COST = {"slippage_pct": 0.0, "commission": 0.0}


def test_close_stop_covers_the_next_open_after_a_close_through_it():
    closes, opens = list(CRASH), list(OPENS)
    closes[76] = 11.6                                    # 26% above the 9.2 short at the next open
    opens[77] = 11.0
    p = panel_from_closes(closes, opens)
    ev = ct.crash_events(p, np.ones(len(closes), dtype=bool))
    xp = {**ss.exit_params(0, 10, 0.0), **NO_COST}
    t = engine_short.run_signals(p, ct.crash_candidates(p, ev, "next_open"), xp)
    t = ct.close_stop(p, t, 25, xp)
    assert t["exit_reason"][0] == "stopped (close)" and t["exit_row"][0] == 77
    assert t["ret_pct"][0] == pytest.approx((9.2 - 11.0) / 9.2 * 100)
    assert t["exit_date"][0] == p.date[77]


def test_close_stop_ignores_an_intraday_squeeze_that_fades():
    highs = list(np.maximum(CRASH, OPENS))
    highs[76] = 12.5                                     # 36% above the short intraday, closes back at 9
    p = panel_from_closes(CRASH, OPENS, highs)
    ev = ct.crash_events(p, np.ones(len(CRASH), dtype=bool))
    xp = {**ss.exit_params(0, 10, 0.0), **NO_COST}
    t = ct.close_stop(p, engine_short.run_signals(p, ct.crash_candidates(p, ev, "next_open"), xp), 25, xp)
    assert t["exit_reason"][0] == "time exit" and t["exit_row"][0] == 83


def test_bounce_entry_is_the_first_open_far_enough_above_the_crash_close():
    opens = list(OPENS)
    opens[74], opens[75] = 10.6, 11.0                    # 11.6% then 15.8% above the 9.5 crash close
    p = panel_from_closes(CRASH, opens)
    ev = ct.crash_events(p, np.ones(len(CRASH), dtype=bool))
    c = ct.crash_candidates(p, ev, "bounce", bounce_pct=10, bounce_bars=5)
    assert list(c["signal_row"]) == [72] and list(c["entry_row"]) == [74]
    assert c["entry_raw"][0] == pytest.approx(10.6) and not c["at_close"][0]
    assert len(ct.crash_candidates(p, ev, "bounce", bounce_pct=20, bounce_bars=5)["entry_row"]) == 0
    assert len(ct.crash_candidates(p, ev, "bounce", bounce_pct=10, bounce_bars=1)["entry_row"]) == 0


def test_parse_stop_and_window_mean():
    assert ct.parse_stop("25") == (25.0, False) and ct.parse_stop("40c") == (40.0, True)
    assert ct.parse_stop("0") == (0.0, False)
    sym = np.array([0] * 5 + [1] * 5)
    x = np.arange(10, dtype=float)
    m = ct.window_mean(x, sym, np.array([4, 2, 6, -1]), 3)
    assert m[0] == pytest.approx(3.0) and m[1] == pytest.approx(1.0)
    assert np.isnan(m[2]) and np.isnan(m[3])            # reaches back into the other stock / before the data
