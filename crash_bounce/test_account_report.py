"""Account report: the per-trade numbers, time in trades and the worst drop."""

import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, HERE)
import account_report as ar  # noqa: E402


def test_trade_stats_in_percent_r_and_dollars():
    t = pd.DataFrame({"ret_pct": [10.0, -5.0, 20.0, -25.0], "bars_held": [10, 10, 10, 2]})
    s = ar.trade_stats(t, years=2.0, capital=10_000, sizes=[5, 10])
    assert s["per_yr"] == 2 and s["win_%"] == 50
    assert s["avg_win_%"] == 15 and s["avg_loss_%"] == -15 and s["payoff"] == 1
    assert s["avg_loss_R"] == pytest.approx(-0.6) and s["worst_R"] == pytest.approx(-1.0)
    assert s["days_held"] == 8
    t["ret_pct"] = [12.0, 12.0, 12.0, -4.0]                # +8% a trade on a $500 and a $1,000 position
    s = ar.trade_stats(t, years=2.0, capital=10_000, sizes=[5, 10])
    assert s["$_a_trade_5%"] == pytest.approx(40) and s["$_a_trade_10%"] == pytest.approx(80)


def test_time_in_trades_counts_days_and_the_share_of_the_account_short():
    curve = pd.DataFrame({"positions": [0, 1, 2, 0], "invested": [0.0, 500.0, 1000.0, 0.0],
                          "equity": [10_000.0] * 4})
    s = ar.time_in_trades(curve)
    assert s["days_in_a_trade_%"] == 50 and s["avg_open"] == 0.75 and s["avg_open_when_in"] == 1.5
    assert s["avg_short_%"] == pytest.approx(3.75) and s["avg_short_%_when_in"] == pytest.approx(7.5)
    assert s["max_short_%"] == pytest.approx(10) and s["avg_cash_%"] == pytest.approx(96.25)


def test_drawdown_is_the_worst_drop_from_a_high():
    curve = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=4), "equity": [100.0, 120.0, 90.0, 130.0]})
    pct, usd, day = ar.drawdown(curve)
    assert pct == pytest.approx(-25) and usd == pytest.approx(30) and day == curve["date"][2]
