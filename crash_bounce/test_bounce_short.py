"""Crash bounce short: the pick spec, the window, and the stop at the close the fall
started from."""

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
