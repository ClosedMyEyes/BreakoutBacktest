"""Spike shorts: which day each setup fires, the first-in-10-bars rule, and the
stop / time exits through the short engine."""

import numpy as np
import pandas as pd
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


def test_min_gap_needs_a_big_enough_gap():
    panel = pn(SPIKE)                              # opens 17 over a 16.5 high: a 3% gap
    assert len(cands(panel, 3, 50, "gap")["entry_row"]) == 1
    keep = np.ones(len(panel.c), dtype=bool)
    c5 = ss.spike_candidates(panel, keep, ss.run_gain(panel, 3), 50, "gap", min_gap=5)
    assert len(c5["entry_row"]) == 0


def test_rsi_simple():
    c = np.array([10, 11, 12, 11, 12.0])
    sym = np.zeros(5, dtype=np.int32)
    r = ss.rsi_simple(c, sym, np.array([4, 2]), n=4)
    assert r[0] == pytest.approx(100 * 3 / 4) and np.isnan(r[1])


def test_reverse_splits_before_counts_only_past_splits_in_the_window():
    bars = [FLAT] * 6
    unadj = [10, 10, 100, 100, 100, 100.0]         # a 1-for-10 reverse split on bar 2
    z = np.zeros(6)
    panel = panel_from(bars, extra={"bad_bar": z, "bad_bar_dn": z, "div": z, "unadj_close": unadj})
    n = ss.reverse_splits_before(panel, np.array([1, 2, 5]))
    assert list(n) == [0, 1, 1]
    assert list(ss.reverse_splits_before(panel, np.array([5]), bars=3)) == [0]


def test_account_curve_marks_open_shorts_and_books_the_trade_when_covered():
    panel = pn([FLAT, FLAT, (10, 10, 8, 8), (8, 9, 8, 9), FLAT])
    dates = pd.DatetimeIndex(np.unique(panel.date))
    t = pd.DataFrame({"entry_row": [1], "exit_row": [3], "entry": [10.0], "ret_pct": [5.0]})
    curve, skipped = ss.account_curve(panel, t, dates, size_pct=10, capital=1000.0)
    # 100 short at 10 = 10 shares: up 20 at the close of 8, booked +5% of 100 when covered
    assert list(curve["equity"]) == pytest.approx([1000, 1000, 1020, 1005, 1005])
    assert list(curve["positions"]) == [0, 1, 1, 0, 0] and skipped == 0


def test_account_curve_skips_trades_with_no_room_and_sizes_on_the_account():
    panel = pn([FLAT] * 6)
    dates = pd.DatetimeIndex(np.unique(panel.date))
    t = pd.DataFrame({"entry_row": [0, 1, 3], "exit_row": [2, 2, 4], "entry": [10.0] * 3,
                      "ret_pct": [10.0, 10.0, 10.0]})
    curve, skipped = ss.account_curve(panel, t, dates, size_pct=50, max_shorts=1, capital=1000.0)
    assert skipped == 1
    assert curve["equity"].iloc[-1] == pytest.approx(1000 * 1.05 * 1.05)


# bar 13 closes at 25, 150% over the 10-bar low of 10; its high is 26
RUN = [FLAT] * 11 + [(12, 16, 12, 15), (15, 21, 15, 20), (20, 26, 20, 25)]


def test_scan_watch_list_gives_the_open_that_triggers_the_short():
    import spike_scan as scan
    panel = pn(RUN)
    w = scan.watch_list(panel, np.ones(len(panel.c), dtype=bool), ss.run_gain(panel, 10))
    assert list(w["symbol"]) == ["AAA"]
    assert w["short_if_open_above"].iloc[0] == pytest.approx(26 * 1.02)


def test_scan_skips_a_stock_that_already_gapped_up_in_the_last_10_bars():
    import spike_scan as scan
    bars = RUN[:-1] + [(22, 26, 21, 25)]          # opened above bar 12's high (21 * 1.02) while in a run
    panel = pn(bars)
    w = scan.watch_list(panel, np.ones(len(panel.c), dtype=bool), ss.run_gain(panel, 10))
    assert w.empty


def test_stop_slip_only_worsens_intraday_stops():
    t = pd.DataFrame({"exit_reason": ["stopped", "stopped (entry day)", "stopped (gap)", "time exit"],
                      "entry": [10.0] * 4, "exit": [12.5, 12.5, 13.0, 9.0], "ret_pct": [-25.0, -25.0, -30.0, 10.0]})
    s = ss.apply_stop_slip(t, 4)
    assert list(s["ret_pct"]) == pytest.approx([-30.0, -30.0, -30.0, 10.0])
    assert list(s["exit"]) == pytest.approx([13.0, 13.0, 13.0, 9.0])


def test_tstat_by_month_counts_a_month_of_trades_once():
    dates = pd.Series(pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05", "2020-04-05",
                                      "2020-05-05", "2020-06-05"]))
    r = pd.Series([1.0, 3.0, 2.0, 4.0, 0.0, 2.0])
    one_each = ss.tstat_by_month(r, dates)
    # the same six returns, each traded four times in its month: no extra evidence
    many = ss.tstat_by_month(pd.concat([r] * 4, ignore_index=True), pd.concat([dates] * 4, ignore_index=True))
    assert one_each == pytest.approx(many)
    e = r - r.mean()
    assert one_each == pytest.approx(r.mean() / np.sqrt((e ** 2).sum() * 6 / 5 / 36))


def test_scan_liquidity_floor_is_the_smallest_dollar_volume_that_passes_on_the_last_day():
    import engine
    import spike_scan as scan
    df = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 3),
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        "listed": [1, 1, 1, 1, 1, 0],
        "dv_pct": [90, 90, 10, 30, 50, 95],         # BBB passes only on the last day; CCC isn't listed then
        "dv50": [1e6, 2e6, 5e3, 3e5, 4e5, 9e3]})
    panel = engine.Panel(df, set())
    assert scan.liquidity_floor(panel, 80.0) == pytest.approx(3e5)
