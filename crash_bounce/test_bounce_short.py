"""Crash bounce short: the pick spec, the window, the stop at the close the fall
started from, and the cover target below the short."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import engine  # noqa: E402
import engine_short  # noqa: E402
import bounce_short as bs  # noqa: E402


def panel_from_closes(closes, opens=None, highs=None, lows=None):
    c = np.asarray(closes, float)
    o = c if opens is None else np.asarray(opens, float)
    h = np.maximum(c, o) if highs is None else np.asarray(highs, float)
    lo = np.minimum(c, o) if lows is None else np.asarray(lows, float)
    n = len(c)
    z = np.zeros(n)
    df = pd.DataFrame({"symbol": "AAA", "date": pd.bdate_range("2020-01-01", periods=n),
                       "open": o, "high": h, "low": lo, "close": c,
                       "volume": 1e6, "sma200": np.nan, "in_sp500": z, "in_sp400": z, "in_sp600": z,
                       "bad_bar": z, "bad_bar_dn": z, "div": z, "unadj_close": c, "dv_pct": 50.0})
    return engine.Panel(df, set())


# 70 bars rising 10 -> 14, then 12, 10.5, 9.5: the 3-day fall from 14 (bar 69) reaches
# 30% on bar 72. r38 = 9.5 + 0.38 * 4.5 = 11.21.
CRASH = list(np.linspace(10, 14, 70)) + [12, 10.5, 9.5] + [9.0] * 15
ARGS = SimpleNamespace(borrow_pct=0.0, slippage_pct=0.0, start=None, end=None, stop_slip=0.0)


@pytest.fixture(autouse=True)
def no_commission(monkeypatch):
    monkeypatch.setitem(engine_short.EXIT_DEFAULTS, "commission", 0.0)


def events(p):
    return bs.ct.crash_events(p, np.ones(len(p.c), dtype=bool))


def test_parse_pick():
    assert bs.parse_pick("r38:touch:5:0") == ("r38", "touch", 5, "0")
    assert bs.parse_pick("prev:open:1:pre") == ("prev", "open", 1, "pre")
    assert bs.parse_pick("10:open:2:25c") == ("10", "open", 2, "25c")
    for bad in ["r38:touch:5", "r38:limit:5:0", "x:open:5:0", "r38:open:0:0", "r38:open:5:pree"]:
        with pytest.raises(ValueError):
            bs.parse_pick(bad)


def test_window_of_one_takes_only_the_day_after_the_crash():
    highs = list(CRASH)
    highs[74] = 11.5                                      # the second day after the crash reaches r38
    p = panel_from_closes(CRASH, highs=highs)
    ev = events(p)
    t, _ = bs.bounce_trades(p, ev, "r38", "touch", 1, "0", 10, ARGS)
    assert t.empty
    t, _ = bs.bounce_trades(p, ev, "r38", "touch", 2, "0", 10, ARGS)
    assert list(t["entry_row"]) == [74] and t["entry_raw"][0] == pytest.approx(9.5 + 0.38 * 4.5)


def test_pre_stop_covers_after_a_close_back_where_the_fall_started():
    closes, highs = list(CRASH), list(CRASH)
    highs[73] = 11.5                                      # shorted at r38 (11.21) the day after the crash
    closes[76] = 14.1                                     # closes above 14, the close the fall started from
    opens = list(closes)
    opens[77] = 13.5
    p = panel_from_closes(closes, opens, np.maximum(highs, closes))
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "pre", 10, ARGS)
    assert t["exit_reason"][0] == "stopped (crash undone)" and t["exit_row"][0] == 77
    short = 9.5 + 0.38 * 4.5
    assert t["ret_pct"][0] == pytest.approx((short - 13.5) / short * 100)


def test_pre_stop_ignores_a_close_short_of_that_line():
    closes, highs = list(CRASH), list(CRASH)
    highs[73] = 11.5
    closes[76] = 13.9                                     # 24% above the short, still under 14
    p = panel_from_closes(closes, highs=np.maximum(highs, closes))
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "pre", 10, ARGS)
    assert t["exit_reason"][0] == "time exit" and t["exit_row"][0] == 83
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "20c", 10, ARGS)
    assert t["exit_reason"][0] == "stopped (close)" and t["exit_row"][0] == 77


def target_case(open76):
    """Shorted at r38 (11.21) on bar 73, the day after the crash; the crash low is 9.5
    (bar 72), so a 0.5x target is 11.21 - 0.5 * 1.71 = 10.355. Bar 73's own low is
    9.0, but it may come before the fill, so it doesn't count. Bar 76 trades down
    to 10.0."""
    closes = list(CRASH)
    closes[73:77] = [11.0, 11.0, 11.0, 10.5]
    opens = list(closes)
    opens[76] = open76
    highs = list(np.maximum(closes, opens))
    highs[73] = 11.5
    lows = list(np.minimum(closes, opens))
    lows[73], lows[76] = 9.0, 10.0
    return panel_from_closes(closes, opens, highs, lows)


SHORT = 9.5 + 0.38 * 4.5
LINE = SHORT - 0.5 * (SHORT - 9.5)


def test_crash_low_is_the_lowest_low_of_the_crash_days():
    lows = list(CRASH)
    lows[71] = 9.1
    assert bs.crash_low(panel_from_closes(CRASH, lows=lows), np.array([72]))[0] == pytest.approx(9.1)
    assert bs.crash_low(panel_from_closes(CRASH), np.array([72]))[0] == pytest.approx(9.5)


def test_target_covers_at_the_line_from_the_day_after_the_entry():
    p = target_case(open76=10.8)
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "0", 10, ARGS, target=0.5)
    assert list(t["entry_row"]) == [73] and t["exit_reason"][0] == "target" and t["exit_row"][0] == 76
    assert t["ret_pct"][0] == pytest.approx((SHORT - LINE) / SHORT * 100)


def test_target_fills_at_the_open_on_a_gap_below_it():
    p = target_case(open76=10.0)
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "0", 10, ARGS, target=0.5)
    assert t["exit_row"][0] == 76 and t["ret_pct"][0] == pytest.approx((SHORT - 10.0) / SHORT * 100)


def test_target_counts_on_the_time_exits_last_day_only():
    p = target_case(open76=10.8)
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "0", 3, ARGS, target=0.5)
    assert t["exit_reason"][0] == "target" and t["exit_row"][0] == 76
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "0", 2, ARGS, target=0.5)
    assert t["exit_reason"][0] == "time exit" and t["exit_row"][0] == 75
    t, _ = bs.bounce_trades(p, events(p), "r38", "touch", 5, "0", 10, ARGS, target=1.5)
    assert t["exit_reason"][0] == "time exit"           # 1.5x is 8.65; it bottoms at 9.0 after the entry


def test_touch_margin_needs_the_high_to_clear_the_line():
    highs = list(CRASH)
    highs[73] = 10.55                                     # 0.5% above prev (10.5), the close before the crash day
    p = panel_from_closes(CRASH, highs=highs)
    t, _ = bs.bounce_trades(p, events(p), "prev", "touch", 1, "0", 10, ARGS)
    assert list(t["entry_row"]) == [73] and t["entry_raw"][0] == pytest.approx(10.5)
    t, _ = bs.bounce_trades(p, events(p), "prev", "touch", 1, "0", 10, ARGS, margin=1.0)
    assert t.empty
    t, _ = bs.bounce_trades(p, events(p), "prev", "touch", 1, "0", 10, ARGS, margin=0.4)
    assert t["entry_raw"][0] == pytest.approx(10.5)        # still filled at the line, not at the high
