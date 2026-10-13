"""Bounce scanner: tomorrow's order at the close before the crash day, in the prices
the stock traded at, and the recent shorts with the days left to the cover."""

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
import bounce_scan as scan  # noqa: E402

# 70 bars rising 10 -> 14, then 12, 10.5, 9.5: the 3-day fall from 14 (bar 69) reaches
# 30% on bar 72, the crash day. The limit is bar 71's close, 10.5.
CRASH = list(np.linspace(10, 14, 70)) + [12, 10.5, 9.5] + [9.0] * 15


def panel(n, highs=None, real=1.0, junk_row=None):
    c = np.asarray(CRASH[:n], float)
    h = c.copy() if highs is None else np.asarray(highs[:n], float)
    z = np.zeros(n)
    bad = z.copy()
    if junk_row is not None:
        bad[junk_row] = 1
    df = pd.DataFrame({"symbol": "AAA", "date": pd.bdate_range("2020-01-01", periods=n),
                       "open": c, "high": h, "low": c, "close": c, "volume": 1e6, "sma200": np.nan,
                       "in_sp500": z, "in_sp400": z, "in_sp600": z, "bad_bar": z, "bad_bar_dn": bad, "div": z,
                       "unadj_close": c * real})
    p = engine.Panel(df, set())
    return p, scan.ct.crash_events(p, np.ones(n, dtype=bool))


@pytest.fixture(autouse=True)
def no_commission(monkeypatch):
    monkeypatch.setitem(engine_short.EXIT_DEFAULTS, "commission", 0.0)


def test_a_crash_on_the_last_bar_is_an_order_at_the_close_before_the_crash_day():
    p, ev = panel(73, real=2.0)                          # prices since halved by a 2-for-1 split
    o = scan.orders(p, ev)
    assert len(o) == 1 and o["crash_date"][0] == p.date[72]
    assert o["short_limit"][0] == pytest.approx(21.0) and o["close"][0] == pytest.approx(19.0)
    assert o["fall_3d_%"][0] == pytest.approx((9.5 / 14 - 1) * 100)
    assert o["limit_over_close_%"][0] == pytest.approx((10.5 / 9.5 - 1) * 100)


def test_no_order_for_an_older_crash_or_a_junk_crash_day():
    assert scan.orders(*panel(74)).empty                 # the crash was yesterday: its day has passed
    assert scan.orders(*panel(73, junk_row=72)).empty    # the backtest skips a crash day that's a price break


def test_recent_trades_show_an_open_short_and_then_its_cover():
    highs = list(CRASH)
    highs[73] = 10.6                                     # the day after the crash reaches the limit
    t = scan.recent_trades(*panel(79, highs), borrow_pct=0.0)
    assert len(t) == 1
    row = t.iloc[0]
    assert row["entry_date"] == pd.Timestamp(pd.bdate_range("2020-01-01", periods=74)[73])
    assert row["filled"] == "at the limit" and row["short_px"] == pytest.approx(10.5)
    assert row["status"] == "open" and row["days_held"] == 5 and row["days_left"] == 5
    t = scan.recent_trades(*panel(88, highs), borrow_pct=0.0)
    assert t.iloc[0]["status"] == "covered" and t.iloc[0]["days_left"] == 0 and t.iloc[0]["days_held"] == 10


def test_no_trade_when_the_day_after_falls_short_of_the_limit():
    highs = list(CRASH)
    highs[73] = 10.4
    assert scan.recent_trades(*panel(79, highs), borrow_pct=0.0).empty


def test_the_record_keeps_one_row_per_order_and_trade_across_nights(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "SIGNALS_FILE", str(tmp_path / "bounce_signals.csv"))
    p, ev = panel(73)
    o = scan.orders(p, ev)
    assert scan.save_signals(o, pd.DataFrame(), pd.Timestamp("2020-04-13"))        # a night with only an order
    highs = list(CRASH)
    highs[73] = 10.6
    t = scan.recent_trades(*panel(79, highs), borrow_pct=0.0)
    for night in ["2020-04-20", "2020-04-21"]:                                       # the short, two nights running
        assert scan.save_signals(pd.DataFrame(), t, pd.Timestamp(night))
    assert not scan.save_signals(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2020-04-22"))
    rec = pd.read_csv(tmp_path / "bounce_signals.csv")
    assert sorted(rec["kind"]) == ["order", "trade"] and rec["crash_date"].nunique() == 1
    assert rec.loc[rec["kind"] == "trade", "scanned"].item() == "2020-04-21"
