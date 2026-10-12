"""Spike shorts: which day each setup fires, the first-in-10-bars rule, and the
stop / time exits through the short engine."""

import numpy as np
import pytest

import engine_short
import spike_short as ss
from tests.test_engine import panel_from

FLAT = (10, 10.2, 9.8, 10)


def pn(bars):
    n = len(bars)
    z = np.zeros(n)
    return panel_from(bars, extra={"bad_bar": z, "bad_bar_dn": z, "div": z, "unadj_close": [b[3] for b in bars]})


def cands(panel, k, x, trigger):
    keep = np.ones(len(panel.c), dtype=bool)
    return ss.spike_candidates(panel, keep, ss.run_gain(panel, k), x, trigger)


# bar 4 closes at 16 (60% over the 3-bar low of 10); bar 5 opens at 17, above bar 4's high (16.5)
SPIKE = [FLAT, FLAT, FLAT, (10, 13, 10, 12.5), (13, 16.5, 12.8, 16), (17, 17.5, 14, 14.5),
         (14, 14.5, 13, 13.5)] + [(13, 13.2, 12.8, 13)] * 12


def test_close_entry_is_the_first_day_the_run_qualifies():
    c = cands(pn(SPIKE), 3, 50, "close")
    assert list(c["entry_row"]) == [4]
    assert c["at_close"][0] and c["entry_raw"][0] == pytest.approx(16)


def test_gap_entry_shorts_the_open_after_the_run_day():
    c = cands(pn(SPIKE), 3, 50, "gap")
    assert list(c["signal_row"]) == [4] and list(c["entry_row"]) == [5]
    assert not c["at_close"][0] and c["entry_raw"][0] == pytest.approx(17)


def test_no_gap_no_gap_entry():
    bars = list(SPIKE)
    bars[5] = (15.5, 16, 14, 14.5)                 # opens under bar 4's high
    assert len(cands(pn(bars), 3, 50, "gap")["entry_row"]) == 0


def test_a_second_day_in_the_same_run_does_not_fire_again():
    bars = list(SPIKE)
    bars[5] = (16, 19, 15.8, 18.5)                 # still 50%+ up the next day
    assert list(cands(pn(bars), 3, 50, "close")["entry_row"]) == [4]


def test_time_exit_and_gap_through_stop():
    panel = pn(SPIKE)
    p = {**ss.exit_params(25, 10, 0.0), "slippage_pct": 0.0, "commission": 0.0}
    t = engine_short.simulate_short(panel, 4, 16.0, True, np.nan, p)
    assert t["exit_reason"] == "time exit" and t["exit_row"] == 14
    assert t["ret_pct"] == pytest.approx((16 - 13) / 16 * 100)

    bars = list(SPIKE)
    bars[5] = (21, 22, 20, 21.5)                   # opens above the 20 stop
    t = engine_short.simulate_short(pn(bars), 4, 16.0, True, np.nan, p)
    assert t["exit_reason"] == "stopped (gap)" and t["exit"] == pytest.approx(21)


def test_no_stop_never_stops():
    bars = list(SPIKE)
    bars[5] = (30, 40, 29, 35)
    p = {**ss.exit_params(0, 10, 0.0), "slippage_pct": 0.0, "commission": 0.0}
    t = engine_short.simulate_short(pn(bars), 4, 16.0, True, np.nan, p)
    assert t["exit_reason"] == "time exit"
