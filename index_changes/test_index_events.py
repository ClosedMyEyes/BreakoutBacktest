"""S&P index changes: which day counts as the change, buyouts, and the
flow-day alignment of the return windows."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import index_events as ie  # noqa: E402


def toy(states, closes, symbol="AAA", start="2020-01-01"):
    n = len(states)
    d = pd.bdate_range(start, periods=n)
    st = np.array(states)
    return pd.DataFrame({
        "symbol": symbol, "date": d, "close": np.asarray(closes, float),
        "unadj_close": np.asarray(closes, float), "volume": np.full(n, 1000.0),
        "bad_bar": 0, "bad_bar_dn": 0,
        "in_sp500": (st == 3).astype(np.int8), "in_sp400": (st == 2).astype(np.int8),
        "in_sp600": (st == 1).astype(np.int8),
    })


def events(df):
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    return df, ie.find_events(df, df["symbol"].cat.codes.to_numpy())


def test_promotion_is_found_on_the_first_day_in_the_new_index():
    df, ev = events(toy([1] * 30 + [2] * 40, np.full(70, 10.0)))
    assert list(ev["row"]) == [30]
    assert ev["transition"][0] == "600->400" and ev["kind"][0] == "add"
    assert not ev["buyout"][0]


def test_a_flicker_after_too_few_bars_is_not_an_event_and_buyouts_are_flagged():
    # 600 for 3 bars only after the start, then none: not stable, so no event at row 3...
    df, ev = events(toy([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0] + [0] * 40, np.full(51, 10.0)))
    assert 10 not in list(ev["row"])          # left 600 after only 2 bars in it
    # ...and a drop to none followed by the end of trading is a buyout
    df, ev = events(toy([3] * 30 + [0] * 5, np.full(35, 10.0)))
    assert ev["transition"][0] == "500->none" and ev["buyout"][0]


def test_post_window_starts_at_the_flow_day_close():
    closes = np.full(60, 10.0)
    closes[29] = 11.0       # flow day (day -1): index funds buy at its close
    closes[30:] = 9.9       # then it fades 10%
    df, ev = events(toy([1] * 30 + [2] * 30, closes))
    spx = pd.DataFrame({"date": df["date"], "spx_close": 100.0})
    ev = ie.add_windows(ev, df, spx)
    assert ev["flow"][0] == pytest.approx(10.0)          # 10 -> 11 on day -1
    assert ev["post10"][0] == pytest.approx(-10.0)       # 11 -> 9.9 from the flow-day close
    assert ev["late10"][0] == pytest.approx(0.0)         # day 0 close onwards is flat
    s = ie.stats(ev, 1.0)
    assert s["trade10"] == pytest.approx(10.0 - ie.round_trip_cost())   # short the add


def test_windows_across_a_price_break_are_dropped():
    df, ev = events(toy([1] * 30 + [2] * 30, np.full(60, 10.0)))
    df.loc[33, "bad_bar"] = 1
    spx = pd.DataFrame({"date": df["date"], "spx_close": 100.0})
    ev = ie.add_windows(ev, df, spx)
    assert np.isnan(ev["post10"][0]) and ev["post1"][0] == pytest.approx(0.0)
