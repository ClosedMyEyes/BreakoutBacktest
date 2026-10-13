"""Crash rebound: which day counts as the event, one event per crash, and the
long-view forward returns."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crash_events as ce  # noqa: E402


def frame(closes, opens=None, bad=None):
    n = len(closes)
    c = np.asarray(closes, float)
    d = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "symbol": pd.Categorical(["AAA"] * n), "date": d, "close": c,
        "open": c if opens is None else np.asarray(opens, float),
        "bad_bar": np.zeros(n) if bad is None else np.asarray(bad, float),
    })


def market(df):
    return pd.DataFrame({"date": df["date"], "spx_open": 100.0, "spx_close": 100.0})


def test_first_day_of_a_crash_is_the_event_and_later_days_are_not():
    # 100 for 5 bars, then 80, 65, 60 (a 3-day fall of 40% by bar 7), then flat
    c = [100] * 5 + [80, 65, 60] + [60] * 15
    codes = np.zeros(len(c), dtype=np.int16)
    ev = ce.fall_events(np.asarray(c, float), codes, 3, 30)
    assert list(np.flatnonzero(ev)) == [6]          # 65 / 100: first day 30%+ down over 3 bars
    ev1 = ce.fall_events(np.asarray(c, float), codes, 1, 15)
    assert list(np.flatnonzero(ev1)) == [5]         # the next 15%+ day is the same crash


def test_one_event_per_crash_within_ten_bars():
    c = [100] * 5 + [70] + [70] * 3 + [45] + [45] * 15   # second 30%+ day only 4 bars later
    codes = np.zeros(len(c), dtype=np.int16)
    ev = ce.fall_events(np.asarray(c, float), codes, 1, 30)
    assert list(np.flatnonzero(ev)) == [5]


def test_forward_returns_from_the_close_and_next_open():
    c = [100] * 5 + [50] + [55] * 30
    o = [100] * 5 + [60] + [52] + [55] * 29
    df = frame(c, o)
    codes = df["symbol"].cat.codes.to_numpy()
    rows = np.array([5])
    f = ce.forward(df, codes, rows, market(df), "close")
    assert f["r10"][0] == pytest.approx(0.10)                 # 50 -> 55
    f = ce.forward(df, codes, rows, market(df), "next_open")
    assert f["r10"][0] == pytest.approx(55 / 52 - 1)


def test_upward_break_drops_the_window_and_delisting_uses_last_close():
    c = [100] * 5 + [50] + [55] * 30
    bad = np.zeros(len(c))
    bad[8] = 1
    df = frame(c, bad=bad)
    codes = df["symbol"].cat.codes.to_numpy()
    f = ce.forward(df, codes, np.array([5]), market(df), "close")
    assert np.isnan(f["r10"][0]) and f["r1"][0] == pytest.approx(0.10)
    df = frame([100] * 5 + [50, 40, 30])                      # stops trading two bars later
    f = ce.forward(df, df["symbol"].cat.codes.to_numpy(), np.array([5]), market(df), "close")
    assert f["r10"][0] == pytest.approx(-0.4) and f["delisted"][0]
