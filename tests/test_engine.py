"""Hand-built price paths for every fill and exit rule in engine.py."""

import numpy as np
import pandas as pd
import pytest

import engine

BASE = {**engine.EXIT_DEFAULTS, "slippage_pct": 0.0, "commission": 0.0}


def panel_from(bars, symbol="AAA", delisted=False, extra=None):
    """bars: list of (open, high, low, close)."""
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    df.insert(0, "date", pd.bdate_range("2020-01-01", periods=len(df)))
    df.insert(0, "symbol", symbol)
    for k, v in (extra or {}).items():
        df[k] = v
    return engine.Panel(df, {symbol} if delisted else set())


def trade(panel, i_entry=1, entry_raw=100.0, at_close=False, struct_stop=np.nan, **p):
    return engine.simulate_trade(panel, i_entry, entry_raw, at_close, struct_stop, {**BASE, **p})


FLAT = (100, 101, 99, 100)


def test_stop_hit_intraday():
    pn = panel_from([FLAT, FLAT, (99, 100, 90, 95)])
    t = trade(pn, exit="x1")
    assert t["exit_reason"] == "stopped"
    assert t["exit"] == pytest.approx(92.5)
    assert t["result_R"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    pn = panel_from([FLAT, FLAT, (88, 89, 85, 86)])
    t = trade(pn, exit="x1")
    assert t["exit_reason"] == "stopped (gap)"
    assert t["exit"] == pytest.approx(88)
    assert t["result_R"] == pytest.approx(-12 / 7.5)


def test_entry_day_stop_counts_as_hit():
    pn = panel_from([FLAT, (100, 101, 92, 100), FLAT])
    t = trade(pn, exit="x1")
    assert t["exit_reason"] == "stopped (entry day)"
    assert t["bars_held"] == 0


@pytest.mark.parametrize("bar,mode,stopped", [
    ((96, 104, 92, 103), "path",  False),   # green: dip to 92 came before the 100 fill
    ((96, 104, 92, 103), "touch", True),
    ((99, 104, 92, 95),  "path",  True),    # red: up through 100 first, then down to 92
    ((96, 104, 90, 92),  "path",  True),    # closed at the stop: hit after the fill either way
])
def test_entry_day_stop_after_intraday_fill(bar, mode, stopped):
    pn = panel_from([FLAT, bar, FLAT, FLAT])
    t = trade(pn, exit="time", time_bars=2, entry_day_stop=mode)
    assert (t["exit_reason"] == "stopped (entry day)") == stopped


def test_entry_at_close_skips_entry_day_stop():
    pn = panel_from([FLAT, (100, 101, 92, 100), FLAT, FLAT])
    t = trade(pn, exit="time", time_bars=2, at_close=True)
    assert t["exit_reason"] == "time exit"


def test_target_and_gap_target():
    pn = panel_from([FLAT, FLAT] + [FLAT] * 20 + [(110, 125, 109, 120)])
    t = trade(pn, exit="x1", target_pct=20)
    assert t["exit_reason"] == "target" and t["exit"] == pytest.approx(120)
    pn = panel_from([FLAT, FLAT] + [FLAT] * 20 + [(130, 131, 129, 130)])
    t = trade(pn, exit="x1", target_pct=20)
    assert t["exit_reason"] == "target (gap)" and t["exit"] == pytest.approx(130)


def test_stop_and_target_same_bar_is_stop():
    pn = panel_from([FLAT, FLAT, (100, 125, 90, 100)])
    t = trade(pn, exit="x1", target_pct=20)
    assert t["exit_reason"] == "stopped"


def test_oneil_fast_gain_holds_eight_weeks():
    # +25% on bar 3 (inside 3 weeks): target ignored, hold until bar 40, then
    # exit at the next open after a close below the 50-day
    bars = [FLAT, FLAT, FLAT, (110, 126, 109, 125)] + [(125, 127, 123, 125)] * 45
    sma50 = np.full(len(bars), 110.0)
    sma50[44] = 130.0                    # close 125 < 130 on bar 44
    pn = panel_from(bars, extra={"sma50": sma50})
    t = trade(pn, exit="x2", target_pct=22.5)
    assert t["exit_reason"] == "close below sma50 after hold"
    assert t["exit_row"] == 45


def test_oneil_slow_gain_takes_target():
    bars = [FLAT, FLAT] + [FLAT] * 20 + [(110, 126, 109, 125)]
    pn = panel_from(bars, extra={"sma50": np.full(len(bars), 90.0)})
    t = trade(pn, exit="x2", target_pct=22.5)
    assert t["exit_reason"] == "target"


def test_trailing_average_exits_next_open():
    bars = [FLAT, FLAT, FLAT, (100, 101, 99, 97), (96, 97, 95, 96)]
    ema = np.full(len(bars), 98.0)
    pn = panel_from(bars, extra={"ema21": ema})
    t = trade(pn, exit="x3", trail="ema21")
    assert t["exit_reason"] == "close below ema21"
    assert t["exit_row"] == 4 and t["exit"] == pytest.approx(96)


def test_delisting_exits_at_last_price():
    bars = [FLAT, FLAT, (95, 96, 94, 95)]
    pn = panel_from(bars, delisted=True, extra={"sma50": np.full(3, 50.0)})
    t = trade(pn, exit="x2", stop_pct=10)
    assert t["exit_reason"] == "delisted" and t["exit"] == pytest.approx(95)


def test_slippage_and_commission_worsen_both_fills():
    pn = panel_from([FLAT, FLAT, FLAT])
    t = engine.simulate_trade(pn, 1, 100.0, False, np.nan,
                              {**BASE, "exit": "time", "time_bars": 1, "slippage_pct": 0.2,
                               "commission": 0.01})
    assert t["entry"] == pytest.approx(100 * 1.002 + 0.01)
    assert t["exit"] == pytest.approx(100 * 0.998 - 0.01)


def test_structural_stop_and_max_risk():
    pn = panel_from([FLAT, FLAT, (99, 100, 94, 95)])
    t = trade(pn, exit="x1", stop_mode="structural", struct_stop=95.0)
    assert t["stop"] == pytest.approx(95) and t["exit_reason"] == "stopped"
    assert trade(pn, exit="x1", stop_mode="structural", struct_stop=70.0) is None   # 30% risk > 15%


def test_buy_stop_fill_and_chase_rules():
    #            signal day        breakout day: opens above pivot 100
    bars = [(95, 100, 94, 99), (102, 106, 101, 105), FLAT]
    pn = panel_from(bars)
    setups = {"setup": np.array([True, False, False]), "pivot": np.array([100.0, np.nan, np.nan]),
              "stop": np.full(3, np.nan)}
    cand = engine.find_entries(pn, setups, {"max_chase": 5.0, "entry_at": "stop", "vol_mult": 0})
    assert cand["entry_raw"][0] == pytest.approx(102)          # open above pivot -> open
    cand = engine.find_entries(pn, setups, {"max_chase": 1.0, "entry_at": "stop", "vol_mult": 0})
    assert len(cand["entry_raw"]) == 0                          # 2% above pivot > 1% chase


def test_buy_stop_needs_high_above_pivot_and_same_symbol():
    df = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB"], "date": pd.bdate_range("2020-01-01", periods=3),
        "open": [99, 99, 200], "high": [100, 100, 210], "low": [98, 98, 190], "close": [99, 99, 205],
    })
    pn = engine.Panel(df, set())
    setups = {"setup": np.array([True, True, False]), "pivot": np.array([100.0, 100.0, np.nan]),
              "stop": np.full(3, np.nan)}
    cand = engine.find_entries(pn, setups, {"max_chase": 5.0, "entry_at": "stop", "vol_mult": 0})
    assert len(cand["entry_row"]) == 0    # AAA never trades above 100; row 1 can't use BBB's bar


def test_entry_at_close_and_next_open():
    bars = [(95, 100, 94, 99), (101, 104, 100, 103), (104, 105, 103, 104), FLAT]
    vol = np.array([1.0, 2.0, 1.0, 1.0])
    pn = panel_from(bars, extra={"vol_ratio": vol})
    setups = {"setup": np.array([True, False, False, False]),
              "pivot": np.array([100.0, np.nan, np.nan, np.nan]), "stop": np.full(4, np.nan)}
    b = engine.find_entries(pn, setups, {"max_chase": 5.0, "entry_at": "close", "vol_mult": 1.5})
    assert b["entry_row"][0] == 1 and b["entry_raw"][0] == pytest.approx(103) and b["at_close"][0]
    c = engine.find_entries(pn, setups, {"max_chase": 5.0, "entry_at": "next_open", "vol_mult": 1.5})
    assert c["entry_row"][0] == 2 and c["entry_raw"][0] == pytest.approx(104)
    none = engine.find_entries(pn, setups, {"max_chase": 5.0, "entry_at": "close", "vol_mult": 2.5})
    assert len(none["entry_row"]) == 0


def test_one_position_per_symbol():
    bars = [(95, 100, 94, 99)] + [(101, 104, 100, 103)] * 10
    pn = panel_from(bars)
    setup = np.zeros(len(bars), dtype=bool)
    setup[[0, 2]] = True
    setups = {"setup": setup, "pivot": np.where(setup, 100.0, np.nan), "stop": np.full(len(bars), np.nan)}
    cand = engine.find_entries(pn, setups, {"max_chase": 5.0, "entry_at": "stop", "vol_mult": 0})
    trades = engine.run_signals(pn, cand, {**BASE, "exit": "time", "time_bars": 5})
    assert len(trades) == 1


def test_portfolio_accounting_and_same_day_exit():
    bars = [FLAT] * 10
    pn = panel_from(bars)
    d = pn.date
    trades = pd.DataFrame([
        # stopped on its entry day: must free the slot the same day
        {"symbol": "AAA", "entry_row": 1, "signal_row": 0, "entry_date": d[1], "exit_date": d[1],
         "entry": 100.0, "stop": 92.5, "exit": 92.5, "result_R": -1.0},
        {"symbol": "AAA", "entry_row": 3, "signal_row": 2, "entry_date": d[3], "exit_date": d[6],
         "entry": 100.0, "stop": 92.5, "exit": 115.0, "result_R": 2.0},
    ])
    curve, taken = engine.run_portfolio(pn, trades, {"capital": 100_000, "max_positions": 1,
                                                     "risk_pct": 1.0, "max_pos_pct": 100,
                                                     "rank_by": "random"})
    assert len(taken) == 2
    shares1 = int(100_000 * 0.01 / 7.5)                 # 1% risk / $7.50 per share
    eq1 = 100_000 - shares1 * 7.5
    shares2 = int(eq1 * 0.01 / 7.5)
    assert taken["shares"].tolist() == [shares1, shares2]
    assert curve["equity"].iloc[-1] == pytest.approx(eq1 + shares2 * 15.0)
    assert curve["positions"].iloc[-1] == 0


def test_oneil_breakeven_variant():
    # +25% on bar 3 starts the hold; then a drop to 99 hits the breakeven stop
    # but not the original 92.5 stop
    bars = [FLAT, FLAT, FLAT, (110, 126, 109, 125), (110, 111, 99, 100), FLAT, FLAT]
    sma50 = np.full(len(bars), 90.0)
    pn = panel_from(bars, extra={"sma50": sma50})
    t = trade(pn, exit="x2", hold_stop="breakeven")
    assert t["exit_reason"] == "stopped (breakeven)"
    assert t["exit"] == pytest.approx(100) and t["result_R"] == pytest.approx(0)
    assert t["stop"] == pytest.approx(92.5)                 # R and sizing keep the initial stop
    t = trade(pn, exit="x2", hold_stop="initial")
    assert t["exit_reason"] != "stopped (breakeven)" and t["exit_row"] == 6


def test_breakeven_not_applied_on_the_bar_the_hold_starts():
    bars = [FLAT, FLAT, (100, 126, 95, 120), (105, 106, 104, 105)]
    pn = panel_from(bars, extra={"sma50": np.full(4, 90.0)})
    t = trade(pn, exit="x2", hold_stop="breakeven")
    assert t["exit_reason"] == "end of data"
