"""Crash-keeps-falling short: short-view price breaks and the squeeze measure."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crash_rebound"))
import crash_events as ce  # noqa: E402
import crash_short as cs  # noqa: E402


def frame(closes, highs=None, opens=None, bad_up=None, bad_dn=None):
    n = len(closes)
    c = np.asarray(closes, float)
    z = np.zeros(n)
    return pd.DataFrame({
        "symbol": pd.Categorical(["AAA"] * n), "date": pd.bdate_range("2020-01-01", periods=n),
        "close": c, "open": c if opens is None else np.asarray(opens, float),
        "high": c if highs is None else np.asarray(highs, float),
        "bad_bar": z if bad_up is None else np.asarray(bad_up, float),
        "bad_bar_dn": z if bad_dn is None else np.asarray(bad_dn, float),
    })


def market(df):
    return pd.DataFrame({"date": df["date"], "spx_open": 100.0, "spx_close": 100.0})


def test_short_view_drops_windows_across_a_downward_break_and_keeps_upward_ones():
    c = [100] * 5 + [60] + [50] * 30
    dn = np.zeros(len(c))
    dn[8] = 1
    df = frame(c, bad_dn=dn)
    codes = df["symbol"].cat.codes.to_numpy()
    f = ce.forward(df, codes, np.array([5]), market(df), "close", side="short")
    assert np.isnan(f["r10"][0]) and f["r1"][0] == pytest.approx(50 / 60 - 1)
    up = np.zeros(len(c))
    up[8] = 1
    df = frame(c, bad_up=up)
    f = ce.forward(df, codes, np.array([5]), market(df), "close", side="short")
    assert f["r10"][0] == pytest.approx(50 / 60 - 1)          # an up break is a real loss: kept


def test_max_up_uses_the_highest_high_after_entry():
    c = [100] * 5 + [60] + [62, 70, 58] + [55] * 20
    h = list(c)
    h[7] = 78                                                    # squeezes to 78 intraday on bar 7
    df = frame(c, highs=h)
    codes = df["symbol"].cat.codes.to_numpy()
    m = cs.max_up(df, codes, np.array([5]), "close", 10)
    assert m[0] == pytest.approx(78 / 60 - 1)
    df = frame(c, highs=h, opens=[100] * 6 + [65] + c[7:])       # short the next open at 65
    m = cs.max_up(df, codes, np.array([5]), "next_open", 10)
    assert m[0] == pytest.approx(78 / 65 - 1)
