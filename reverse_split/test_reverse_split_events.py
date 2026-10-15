"""Reverse splits: which days count, the universe on day 0, and the short's sums."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reverse_split_events as rs  # noqa: E402


def toy(symbol, n=300, price=4.0):
    """A flat stock: adjusted and printed close both at `price`, in the universe."""
    c = np.full(n, price)
    return pd.DataFrame({"symbol": symbol, "date": pd.bdate_range("2012-01-02", periods=n), "open": c.copy(),
                         "high": c.copy(), "low": c.copy(), "close": c.copy(), "unadj_close": c.copy(),
                         "volume": 1e6, "turnover": 4e6, "div": 0.0, "listed": 1, "dv_pct": 90.0, "tight_10": 0.1,
                         "bars_since_break": np.nan, "bad_bar": 0, "bad_bar_dn": 0, "dv50": 4e6, "sma200": price,
                         "sma50": price, "hi52_c": price, "in_sp500": 0, "in_sp400": 0, "in_sp600": 0, "in_r2000": 0})


def reverse_split(df, row, ratio):
    """A 1-for-`ratio` reverse split effective at `row`: the printed price before it was
    1/ratio of the adjusted one (the adjusted series stays smooth)."""
    sym = df.loc[row, "symbol"]
    before = (df.index < row) & (df["symbol"] == sym)
    df.loc[before, "unadj_close"] /= ratio


def panel(*frames):
    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    return df, df["symbol"].cat.codes.to_numpy()


def start_of(df, sym):
    return int(df.index[df["symbol"] == sym][0])


def market(df):
    d = pd.DatetimeIndex(np.sort(df["date"].unique()))
    return pd.DataFrame({k: 100.0 for k in ["spx_open", "spx_close", "comp_open", "comp_close"]}, index=d)


def market_features(df, above200=1.0):
    """The market features by date, as datastore.load_market() gives them."""
    d = pd.DatetimeIndex(np.sort(df["date"].unique()))
    return pd.DataFrame({"spx_above200": above200, "spx_above50": 1.0, "spx_sma50_rising": 1.0,
                         "comp_above200": 0.0, "pct_above200": 55.0}, index=d)


P = dict(rs.rules.UNIVERSE_DEFAULTS, min_price=1.0, liq_top_pct=80.0)
ARGS = argparse.Namespace(min_ratio=rs.MIN_RATIO, dedupe=rs.DEDUPE, start=None, end=None)


def events(df, codes):
    ev, _ = rs.select(df, codes, P, ARGS)
    return ev.sort_values("row").reset_index(drop=True)


def test_a_1_for_10_is_found_with_ratio_10_and_a_forward_split_is_not():
    a, b = toy("REV"), toy("FWD")
    df, codes = panel(a, b)
    reverse_split(df, start_of(df, "REV") + 80, 10)
    fwd = (df["symbol"] == "FWD") & (df.index < start_of(df, "FWD") + 80)
    df.loc[fwd, "unadj_close"] *= 2                                 # a 2-for-1 forward split
    ev, counts = rs.find_splits(df, codes)
    assert list(ev["row"]) == [start_of(df, "REV") + 80]
    assert np.isclose(ev["ratio"].iloc[0], 10.0) and not ev["prior_split"].iloc[0]
    assert counts["upward factor jumps on listed bars"] == 1


def test_repeats_short_history_and_bad_prints_are_left_out():
    df, codes = panel(toy("TWO"), toy("LTR"), toy("NEW"), toy("BAD"))
    reverse_split(df, start_of(df, "TWO") + 80, 10)
    reverse_split(df, start_of(df, "TWO") + 110, 5)                  # 30 bars later: a repeat
    reverse_split(df, start_of(df, "LTR") + 80, 10)
    reverse_split(df, start_of(df, "LTR") + 200, 5)                  # 120 bars later: its own event
    reverse_split(df, start_of(df, "NEW") + 30, 10)                  # under 60 bars of history
    spike = start_of(df, "BAD") + 90
    df.loc[spike, "unadj_close"] *= 10                               # one bad print, undone the next bar
    ev, counts = rs.find_splits(df, codes)
    sym = df["symbol"].astype(str).to_numpy()[ev["row"]]
    got = sorted((s, r - start_of(df, s)) for s, r in zip(sym, ev["row"]))
    assert got == [("LTR", 80), ("LTR", 200), ("TWO", 80)]
    assert counts["repeats within the dedupe bars"] == 1
    assert counts["under 60 bars of history"] == 1
    assert counts["bad prints (undone within 5 bars)"] == 1
    later = ev[ev["row"] == start_of(df, "LTR") + 200]
    assert later["prior_split"].iloc[0] and np.isclose(later["ratio"].iloc[0], 5.0)


def test_a_break_on_day_0_is_allowed_but_an_earlier_one_blocks():
    df, codes = panel(toy("DAY0"), toy("EARLY"), toy("CHEAP"))
    for s in ["DAY0", "EARLY", "CHEAP"]:
        reverse_split(df, start_of(df, s) + 80, 10)
    d0 = start_of(df, "DAY0") + 80
    df.loc[d0, ["bad_bar_dn", "bars_since_break"]] = [1, 0]             # the split day fell 25%+
    e = start_of(df, "EARLY")
    df.loc[e + 60:e + 299, "bars_since_break"] = np.arange(240)         # a break 20 bars before day 0
    df.loc[df["symbol"] == "CHEAP", ["unadj_close"]] /= 5               # prints $0.80 on day 0
    ev = events(df, codes)
    assert list(ev["symbol"]) == ["DAY0"]
    ev = rs.features(ev, df, pd.Series(1, index=pd.DatetimeIndex(df["date"].unique())))
    assert bool(ev["break0"].iloc[0]) and np.isclose(ev["px_prev"].iloc[0], 0.4) and np.isclose(ev["px0"].iloc[0], 4)
    assert ev["index"].iloc[0] == "neither" and ev["in_index"].iloc[0] == 0.0


def test_the_short_at_each_fee_and_the_break_even_fee():
    df, codes = panel(toy("FALL"))
    r = 80
    reverse_split(df, r, 10)
    df.loc[r + 1:, ["open", "high", "low", "close"]] = 3.0               # -25% from the day 0 close
    df.loc[r + 1, ["open", "high"]] = 3.6
    ev = events(df, codes)
    mkt = market(df)
    out = rs.trades(ev, df, mkt, "close0", 20)
    g = pd.concat([ev, pd.DataFrame(out)], axis=1)
    cost = rs.ce.round_trip_cost()
    assert np.isclose(g["g_close0:20"].iloc[0], 25.0 - cost)
    for fee in [5, 25, 50, 100, 200]:
        assert np.isclose(rs.ret(g, "close0:20", fee).iloc[0], 25.0 - cost - fee * 20 / 252)
    assert np.isclose(rs.ret(g, "close0:20", 50, "h").iloc[0], 25.0 - cost - 50 * 20 / 252)   # flat S&P
    assert np.isclose(rs.breakeven(g, "close0:20"), (25.0 - cost) / (20 / 252))
    assert np.isclose(rs.ret(g, "close0:20", rs.breakeven(g, "close0:20")).iloc[0], 0.0)
    op = rs.trades(ev, df, mkt, "open1", 20)
    assert np.isclose(op["g_open1:20"][0], -(3.0 / 3.6 - 1) * 100 - cost)
    assert op["exit_open1:20"][0] == out["exit_close0:20"][0] == r + 20
    late = rs.trades(ev, df, mkt, "close5", 20)
    assert late["exit_close5:20"][0] == r + 25 and np.isclose(late["g_close5:20"][0], -cost)


def test_a_delisted_stock_is_covered_at_its_last_close_and_one_past_the_data_end_is_left_out():
    gone, live = toy("GONE", n=100), toy("LIVE", n=300)
    df, codes = panel(gone, live)
    g0, l0 = start_of(df, "GONE"), start_of(df, "LIVE")
    reverse_split(df, g0 + 80, 10)
    reverse_split(df, l0 + 280, 10)
    df.loc[g0 + 81:g0 + 99, ["open", "high", "low", "close"]] = 2.0     # then stops trading at row 99
    ev = events(df, codes)
    assert list(ev["gone"]) == [True, False]
    out = rs.trades(ev, df, market(df), "close0", 60)
    cost = rs.ce.round_trip_cost()
    assert np.isclose(out["g_close0:60"][0], 50.0 - cost)              # 4.0 -> 2.0, covered at its last close
    assert np.isclose(out["y_close0:60"][0], 19 / 252) and out["del_close0:60"][0] == 1.0
    assert out["exit_close0:60"][0] == g0 + 99
    assert np.isnan(out["g_close0:60"][1])                             # still trading when the data ends


def test_a_downward_break_in_the_hold_is_left_out_an_upward_one_counts_and_squeezes_are_counted():
    df, codes = panel(toy("DN"), toy("UP"), toy("SQZ"))
    for s in ["DN", "UP", "SQZ"]:
        reverse_split(df, start_of(df, s) + 80, 10)
    dn, up, sq = (start_of(df, s) + 80 for s in ["DN", "UP", "SQZ"])
    df.loc[dn + 5:dn + 219, ["open", "high", "low", "close"]] = 1.0
    df.loc[dn + 5, "bad_bar_dn"] = 1                                   # an untrusted gain for a short
    df.loc[up + 5:up + 219, ["open", "high", "low", "close"]] = 6.0
    df.loc[up + 5, "bad_bar"] = 1                                      # a real loss
    df.loc[sq + 3, "high"] = 6.2                                       # 55% above the entry, then back
    ev = events(df, codes)
    out = rs.trades(ev, df, market(df), "close0", 20)
    g = dict(zip(ev["symbol"], out["g_close0:20"]))
    adv = dict(zip(ev["symbol"], out["adv_close0:20"]))
    cost = rs.ce.round_trip_cost()
    assert np.isnan(g["DN"])
    assert np.isclose(g["UP"], -50.0 - cost)
    assert np.isclose(g["SQZ"], -cost) and adv["SQZ"] == 1.0 and adv["UP"] == 1.0


def test_profile_counts_the_drift_after_day_0():
    df, codes = panel(toy("DRIFT"))
    r = 80
    reverse_split(df, r, 10)
    df.loc[r + 1:, "close"] = 4.0 * 0.99 ** np.arange(1, len(df) - r)   # -1% a day after day 0
    ev = events(df, codes)
    prof = rs.profile(ev, df, market(df)).set_index("day")
    assert np.isclose(prof.loc[1, "vs_spx"], 1.0) and np.isclose(prof.loc[1, "running"], 1.0)
    assert np.isclose(prof.loc[10, "running"], 10.0)
    assert prof.loc[-1, "under1_%"] == 100.0 and prof.loc[0, "under1_%"] == 0.0


def test_parse_pick():
    assert rs.parse_pick("close0:60") == ("close0", 60)
    assert rs.parse_pick("open1:5") == ("open1", 5)
    for bad in ["close1:5", "close0:0", "close0"]:
        try:
            rs.parse_pick(bad)
            assert False, bad
        except ValueError:
            pass


def test_prior_n_counts_the_reverse_splits_in_the_3_years_before():
    df, codes = panel(toy("MANY", n=1000))
    for row in [80, 200, 320, 900]:                                  # 900 - 756 = 144: 80 is too old
        reverse_split(df, row, 5)
    ev, _ = rs.find_splits(df, codes)
    assert list(ev["row"]) == [80, 200, 320, 900]
    assert list(ev["prior_n"]) == [0, 1, 2, 2]
    assert list(ev["prior_split"]) == [False, True, True, True]


def test_window_mean_stays_inside_the_stock():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    got = rs.window_mean(x, np.array([5, 5, 3]), -3, -1, np.array([0, 3, 0]))
    assert np.isclose(got[0], 4.0)                                   # rows 2-4, the NaN skipped
    assert np.isnan(got[1])                                          # row 2 is before the stock's first row
    assert np.isclose(got[2], 2.0)                                   # rows 0-2


def test_the_explore_features_downtrend_volume_and_market():
    df, codes = panel(toy("DOWN", n=400))
    r = 300
    px = ["open", "high", "low", "close", "unadj_close"]
    df.loc[:r - 251, px] = 16.0                                       # 16 until 250 bars before day 0
    df.loc[r - 250:r - 121, px] = 8.0                                 # then 8, then 6 until 60 bars before
    df.loc[r - 120:r - 61, px] = 6.0
    df["hi52_c"] = 6.0
    reverse_split(df, r, 10)
    df["turnover"] = df["close"] * 1e6                               # 1M shares a day
    df.loc[r - 10:r - 1, "turnover"] = 4.0 * 0.5e6                   # half the usual shares in the last 10 days
    df.loc[r, "turnover"] = 4.0 * 3e6                                # 3x on day 0
    df.loc[r, "sma50"] = 5.0                                         # below its 50-day
    ev = events(df, codes)
    ev = rs.features(ev, df, pd.Series(1.0, index=pd.DatetimeIndex(df["date"].unique())), market_features(df),
                     market(df))
    e = ev.iloc[0]
    assert np.isclose(e["run60"], 0.0) and np.isclose(e["move120"], -100 / 3) and np.isclose(e["move250"], -50.0)
    assert np.isclose(e["from_high"], 100 / 3) and e["above50"] == 0.0
    assert np.isclose(e["vol10_x"], 0.5) and np.isclose(e["vol0_x"], 3.0)
    assert e["spx_trend"] == 1.0 and e["comp_above200"] == 0.0 and e["breadth"] == 55.0 and e["spx_move20"] == 0.0


def test_a_stop_covers_at_the_stop_or_the_open_above_it():
    df, codes = panel(toy("TOUCH"), toy("GAP"), toy("NONE"), toy("LATE"))
    for s in ["TOUCH", "GAP", "NONE"]:
        reverse_split(df, start_of(df, s) + 80, 10)
    reverse_split(df, start_of(df, "LATE") + 280, 10)                # 60 bars runs past the data end
    t, gp, lt = (start_of(df, s) for s in ["TOUCH", "GAP", "LATE"])
    df.loc[t + 83, "high"] = 6.2                                     # touches +50% (6.0) on day 3
    df.loc[gp + 83, ["open", "high", "close"]] = [7.0, 7.5, 7.0]     # opens above it
    df.loc[lt + 285, "high"] = 6.5
    ev = events(df, codes)
    out = rs.trades(ev, df, market(df), "close0", 60, 50)
    got = {s: k for k, s in enumerate(ev["symbol"])}
    g, stop, ex = out["g_close0:60/s50"], out["stop_close0:60/s50"], out["exit_close0:60/s50"]
    cost = rs.ce.round_trip_cost()
    assert np.isclose(g[got["TOUCH"]], -50.0 - cost) and ex[got["TOUCH"]] == t + 83
    assert np.isclose(g[got["GAP"]], -75.0 - cost) and ex[got["GAP"]] == gp + 83
    assert np.isclose(g[got["NONE"]], -cost) and stop[got["NONE"]] == 0.0
    assert stop[got["TOUCH"]] == 1.0 and np.isclose(out["y_close0:60/s50"][got["TOUCH"]], 3 / 252)
    assert np.isclose(g[got["LATE"]], -50.0 - cost) and ex[got["LATE"]] == lt + 285   # stopped before the end
    op = rs.trades(ev, df, market(df), "open1", 60, 50)
    assert np.isclose(op["g_open1:60/s50"][got["TOUCH"]], -50.0 - cost)
    df.loc[t + 81, "high"] = 6.1                                     # the open1 entry bar itself
    op = rs.trades(ev, df, market(df), "open1", 60, 50)
    assert np.isclose(op["g_open1:60/s50"][got["TOUCH"]], -50.0 - cost)
    assert op["exit_open1:60/s50"][got["TOUCH"]] == t + 81


def test_explore_prints_every_family_and_counts_its_tries(capsys):
    frames = [toy(f"S{k}", n=400) for k in range(6)]
    df, codes = panel(*frames)
    for k in range(6):
        s0 = start_of(df, f"S{k}")
        reverse_split(df, s0 + 80 + 10 * k, [5, 10, 30][k % 3])
        df.loc[s0 + 81 + 10 * k:s0 + 399, ["open", "high", "low", "close"]] = 4.0 * (0.7 if k % 2 else 1.2)
    ev = events(df, codes)
    mkt = market(df)
    ev = rs.features(ev, df, pd.Series(1.0, index=pd.DatetimeIndex(df["date"].unique())), market_features(df), mkt)
    grid = [(e, hh) for e in rs.ENTRIES for hh in rs.HOLDS]
    cols = {}
    for e, hh in grid:
        cols.update(rs.trades(ev, df, mkt, e, hh))
    ev = pd.concat([ev, pd.DataFrame(cols)], axis=1)
    pick = ev[ev["g_close0:60"].notna()].reset_index(drop=True)
    args = argparse.Namespace(pick="close0:60", min_ratio=rs.MIN_RATIO)
    tried = rs.explore(pick, ev, df, mkt, "close0:60", 50.0, grid, args)
    n_buckets = sum(len(b) for _, filters in rs.explore_groups(pick, rs.MIN_RATIO) for _, b in filters)
    assert tried == n_buckets + len(grid) + len(rs.STOPS)
    text = capsys.readouterr().out
    for head in ["DOWNTREND", "MARKET", "VOLUME", "THE SPLIT", "SIZE", "EVERY ENTRY AND HOLD", "STOPS"]:
        assert head in text
    assert "2010-14" in text and "acct2_maxdd" in text


def test_frozen_sets_the_pick_and_filter_and_refuses_overrides():
    ns = lambda **kw: argparse.Namespace(**{"frozen": None, "pick": None, "only": [], **kw})
    assert rs.apply_frozen(ns()).pick == rs.PICK
    a = rs.apply_frozen(ns(frozen="0-20"))
    assert (a.pick, a.only) == ("close0:20", ["ratio:20:inf", "in_index:0:1"])
    a = rs.apply_frozen(ns(frozen="20-80"))
    assert (a.pick, a.only) == ("close20:60", ["ratio:20:inf", "in_index:0:1", "off_peak:25:inf", "dv_m:1:inf",
                                               "px_e:1:inf", "spx_above200_e:1:2"])
    assert a.only is not rs.FROZEN["20-80"]["only"]
    for bad in [ns(frozen="0-20", pick="close0:60"), ns(frozen="20-80", only=["ratio:5:20"])]:
        try:
            rs.apply_frozen(bad)
            assert False
        except SystemExit:
            pass


def test_filter_table_and_the_sizing_check():
    n = 8
    g = pd.DataFrame({"date": pd.bdate_range("2016-01-04", periods=n, freq="35D"),
                      "index": ["neither"] * 6 + ["Russell 2000 only"] * 2, "spx_move20": 0.0, "spx_above200": 1.0,
                      "dv_m": [2.0] * 4 + [0.5] * 4, "vol50_x": 2.0, "from_high": 95.0, "vol0_x": 3.0,
                      "move250": -95.0, "move120": -80.0, "prior_n": [1, 2, 1, 1, 0, 0, 0, 0]})
    name = "close0:20"
    g[f"g_{name}"] = [30.0, 32.0, 28.0, 30.0, 10.0, 12.0, -20.0, -22.0]
    for k in ["h", "n"]:
        g[f"{k}_{name}"] = g[f"g_{name}"]
    g[f"y_{name}"], g[f"adv_{name}"], g[f"del_{name}"] = 0.0, 0.0, 0.0      # no borrow, to keep the sums simple
    g[f"peak_{name}"] = [10.0] * 7 + [250.0]                              # one ran to 3.5x the entry
    t = rs.filter_table(g, name, 50.0).set_index("filter")
    assert t.loc["all trades", "trades"] == 8 and np.isclose(t.loc["all trades", "total"], 100.0)
    dv = t.loc["$1M+ a day traded"]
    assert dv["trades"] == 4 and np.isclose(dv["ret"], 30.0) and np.isclose(dv["ret_dropped"], -5.0)
    assert np.isclose(dv["total_vs_all_%"], 120.0)
    row = rs.block_row(g, name, 50.0)
    assert np.isclose(row["x3_%"], 12.5) and np.isclose(row["worst"], -22.0)
    na, ra, nb, rb, diff, tw = rs.sizing_check(g, name, 50.0)
    assert (na, nb) == (4, 4) and np.isclose(ra, 30.0) and np.isclose(rb, -5.0) and np.isclose(diff, 35.0)
    assert tw > 2


def test_account_sized_matches_the_account_and_opens_a_short_at_its_scale():
    df, codes = panel(toy("DOWN"), toy("UP"))
    for s in ["DOWN", "UP"]:
        reverse_split(df, start_of(df, s) + 80, 10)
    df.loc[start_of(df, "DOWN") + 81:start_of(df, "UP") - 1, ["open", "high", "low", "close"]] = 3.0   # -25%
    df.loc[start_of(df, "UP") + 81:, ["open", "high", "low", "close"]] = 6.0                           # +50%
    ev = events(df, codes)
    g = pd.concat([ev, pd.DataFrame(rs.trades(ev, df, market(df), "close0", 20))], axis=1)
    g["short_0:20"] = rs.ret(g, "close0:20", 50)
    spx = market(df)["spx_close"]
    full, _ = rs.le.account(g, df, spx, 0, 20, 2.0, 50)
    same, _ = rs.account_sized(g, df, spx, 0, 20, 2.0, 50, np.ones(len(g)))
    pd.testing.assert_frame_equal(full, same)
    up = (g["symbol"] == "UP").to_numpy()
    half, _ = rs.account_sized(g, df, spx, 0, 20, 2.0, 50, np.where(up, 0.5, 1.0))
    r = g["short_0:20"].to_numpy()
    assert np.isclose(r[~up][0], 25.0 - rs.ce.round_trip_cost() - 50 * 20 / 252)
    assert np.isclose(half["equity"].iloc[-1], 1 + 0.02 * r[~up][0] / 100 + 0.01 * r[up][0] / 100)
    assert half["equity"].iloc[-1] > full["equity"].iloc[-1]


def test_entry_features_at_a_later_close():
    df, codes = panel(toy("E"), toy("LATE"))
    r, late = start_of(df, "E") + 80, start_of(df, "LATE") + 290
    reverse_split(df, r, 10)
    reverse_split(df, late, 10)
    df.loc[r + 5, "high"] = 6.0                                      # +50% above the day 0 close on day 5
    df.loc[r + 20, ["close", "unadj_close"]] = 3.0                   # -25% by the day 20 close
    df.loc[r + 1:r + 20, "turnover"] = df.loc[r + 1:r + 20, "close"] * 2e6   # twice the usual shares
    ev = rs.entry_features(events(df, codes), df, 20, market_features(df))
    e, x = ev[ev["symbol"] == "E"].iloc[0], ev[ev["symbol"] == "LATE"].iloc[0]
    assert np.isclose(e["since0"], -25.0) and np.isclose(e["peak_since0"], 50.0) and np.isclose(e["off_peak"], 50.0)
    assert e["px_e"] == 3.0 and np.isclose(e["dv_e_m"], 4.0) and np.isclose(e["vol_since_x"], 2.0)
    assert e["spx_above200_e"] == 1.0
    assert all(np.isnan(x[k]) for k in ["since0", "peak_since0", "px_e", "vol_since_x", "spx_above200_e"])


def test_explore_with_a_day_20_entry_adds_the_entry_family(capsys):
    frames = [toy(f"S{k}", n=400) for k in range(6)]
    df, codes = panel(*frames)
    for k in range(6):
        s0 = start_of(df, f"S{k}")
        reverse_split(df, s0 + 80 + 10 * k, [5, 10, 30][k % 3])
        df.loc[s0 + 101 + 10 * k:s0 + 399, ["open", "high", "low", "close"]] = 4.0 * (0.7 if k % 2 else 1.2)
    ev = events(df, codes)
    mkt = market(df)
    ev = rs.features(ev, df, pd.Series(1.0, index=pd.DatetimeIndex(df["date"].unique())), market_features(df), mkt)
    ev = rs.entry_features(ev, df, 20, market_features(df))
    grid = [(e, hh) for e in rs.ENTRIES for hh in rs.HOLDS]
    cols = {}
    for e, hh in grid:
        cols.update(rs.trades(ev, df, mkt, e, hh))
    ev = pd.concat([ev, pd.DataFrame(cols)], axis=1)
    pick = ev[ev["g_close20:60"].notna()].reset_index(drop=True)
    args = argparse.Namespace(pick="close20:60", min_ratio=rs.MIN_RATIO)
    tried = rs.explore(pick, ev, df, mkt, "close20:60", 50.0, grid, args)
    n_buckets = sum(len(b) for _, filters in rs.explore_groups(pick, rs.MIN_RATIO, 20) for _, b in filters)
    assert tried == n_buckets + len(grid) + len(rs.STOPS)
    assert "AT THE ENTRY (the day 20 close)" in capsys.readouterr().out


def test_a_filter_at_the_entry_close_hides_the_earlier_entries():
    grid = [(e, hh) for e in rs.ENTRIES for hh in rs.HOLDS]
    assert rs.entry_grid(grid, "close20", ["ratio:20:inf"]) == grid
    assert rs.entry_grid(grid, "close20", ["ratio:20:inf", "off_peak:25:inf"]) == [("close20", hh) for hh in rs.HOLDS]
    assert {e for e, _ in rs.entry_grid(grid, "close5", ["px_e:1:inf"])} == {"close5", "close20"}


def test_left_out_keeps_what_every_other_filter_keeps_and_this_one_drops():
    ev = pd.DataFrame({"ratio": [30, 30, 30, 10], "spx_above200_e": [1.0, 0.0, np.nan, 0.0]})
    out = rs.left_out(ev, ["ratio:20:inf", "spx_above200_e:1:2"], "spx_above200_e:1:2")
    assert out["spx_above200_e"].isna().tolist() == [False, True] and out["spx_above200_e"].iloc[0] == 0.0
    assert len(rs.keep_only(ev, ["ratio:20:inf", "spx_above200_e:1:2"])) + len(out) == 3


def test_combined_puts_both_frozen_rules_in_one_account(capsys):
    df, codes = panel(toy("S"))
    r = start_of(df, "S") + 80
    reverse_split(df, r, 30)
    df.loc[r + 5, "high"] = 6.0                                                    # the post-split high
    df.loc[r + 6:, ["open", "high", "low", "close", "unadj_close"]] = 3.0          # 50% below it from day 6 on
    mkt = market(df)
    ev = rs.features(events(df, codes), df, pd.Series(1.0, index=pd.DatetimeIndex(df["date"].unique())),
                     market_features(df), mkt)
    out = rs.combined(rs.entry_features(ev, df, 20, market_features(df)), df, mkt, 50.0, [2.0])
    assert out["rules"].tolist() == ["0-20", "20-80", "both"] and out["trades"].tolist() == [1, 1, 2]
    cost = rs.ce.round_trip_cost()
    r1, r2 = 25.0 - cost - 50 * 20 / 252, -cost - 50 * 60 / 252                   # day 0 to 20, then flat to 80
    assert np.isclose(out["ret"].iloc[0], r1) and np.isclose(out["ret"].iloc[1], r2)
    assert np.isclose(out["x_start"].iloc[2], 1 + 0.02 * r1 / 100 + 0.02 * (1 + 0.02 * r1 / 100) * r2 / 100)
    assert "1 of the 1 20-80 shorts follow a 0-20 short" in capsys.readouterr().out


def test_next_open_shorts_after_the_close_it_is_decided_on_and_mark_open_keeps_a_trade_at_the_data_end():
    df, codes = panel(toy("A", n=400), toy("LIVE", n=400))
    r, live = start_of(df, "A") + 80, start_of(df, "LIVE") + 380
    reverse_split(df, r, 30)
    reverse_split(df, live, 30)
    df.loc[r + 21, "open"] = 5.0                                      # the day 21 open, above the day 20 close
    df.loc[live + 5:live + 19, ["open", "high", "low", "close"]] = 3.0
    ev = events(df, codes)
    mkt = market(df)
    a, b = rs.trades(ev, df, mkt, "close0", 20, next_open=True), rs.trades(ev, df, mkt, "open1", 20)
    assert all(np.allclose(a[k], b[k.replace("close0", "open1")], equal_nan=True) for k in a)
    x = rs.trades(ev, df, mkt, "close20", 60, next_open=True)
    assert x["exit_close20:60"][0] == r + 80 and x["px_close20:60"][0] == 5.0
    assert np.isclose(x["g_close20:60"][0], -(4.0 / 5.0 - 1) * 100 - rs.ce.round_trip_cost())
    assert np.isclose(x["y_close20:60"][0], 60 / 252)
    run, marked = rs.trades(ev, df, mkt, "close0", 60), rs.trades(ev, df, mkt, "close0", 60, mark_open=True)
    assert np.isnan(run["g_close0:60"][1]) and marked["exit_close0:60"][1] == live + 19
    assert np.isclose(marked["g_close0:60"][1], 25.0 - rs.ce.round_trip_cost())
    assert np.isclose(marked["y_close0:60"][1], 19 / 252) and marked["del_close0:60"][1] == 0.0
    assert np.isclose(marked["g_close0:60"][0], run["g_close0:60"][0])


def test_real_costs_charge_the_commission_at_the_printed_price_dividends_and_every_market_day():
    df, codes = panel(toy("S"))
    r = start_of(df, "S") + 80
    reverse_split(df, r, 30)
    df.loc[r + 5, "div"] = 0.04                                       # 1% of the $4 price, ex on day 5
    df.loc[r + 25, "div"] = 0.04                                      # after the cover: not charged
    mkt = market(df)
    gap = df.index[r + 10]
    df = df.drop(index=gap).reset_index(drop=True)                    # no bar on one market day in the hold
    codes = df["symbol"].astype("category").cat.codes.to_numpy()
    ev = rs.features(events(df, codes), df, pd.Series(1.0, index=mkt.index), market_features(df), mkt)
    run = rs.frozen_trades(ev, df, mkt, "0-20", 50.0)
    real = rs.frozen_trades(ev, df, mkt, "0-20", 50.0, real_cost=True)
    cost = 2 * rs.AUDIT_SLIP + 2 * rs.config.COMMISSION_PER_SHARE / 4.0 * 100
    assert np.isclose(real["cost_%"].iloc[0], cost) and np.isclose(real["div_%"].iloc[0], 1.0)
    g = -rs.ce.round_trip_cost()                                      # flat, so the short is just its cost
    assert np.isclose(run["short_0:1000000"].iloc[0], g - 50 * 20 / 252)
    assert np.isclose(real["short_0:1000000"].iloc[0], -cost - 1.0 - 50 * 21 / 252)


def test_a_short_at_the_open_is_sized_on_the_account_at_the_close_before():
    df, codes = panel(toy("A"), toy("B"))
    a0, b0 = start_of(df, "A"), start_of(df, "B")
    df.loc[a0 + 101:a0 + 299, ["open", "high", "low", "close"]] = 3.0   # A's short gains 25% on day 101
    spx = market(df)["spx_close"]
    g = pd.DataFrame({"row": [a0 + 100, b0 + 101], "last": [a0 + 120, b0 + 120], "entry_px": [4.0, 4.0],
                      "at_open": [False, True], "short_0:5": [25.0, 0.0]})      # B opens on day 101, flat
    alone, _ = rs.account_sized(g.iloc[:1], df, spx, 0, 5, 2.0, 0.0, np.ones(1))
    at_open, _ = rs.account_sized(g, df, spx, 0, 5, 2.0, 0.0, np.ones(2))
    at_close, _ = rs.account_sized(g.assign(at_open=False), df, spx, 0, 5, 2.0, 0.0, np.ones(2))
    mark = -rs.ce.round_trip_cost() / 2 / 100                          # B's mark while open: half the round trip
    for curve, sized_on in [(at_open, alone["equity"].iloc[0]), (at_close, alone["equity"].iloc[1])]:
        assert np.isclose(curve["equity"].iloc[2] - alone["equity"].iloc[2], 0.02 * sized_on * mark)
    assert np.isclose(at_open["equity"].iloc[-1], 1 + 0.02 * 25.0 / 100)   # B covers at 0, whatever its size


def test_audit_runs_each_frozen_rule_as_run_and_with_each_shortcut_out(capsys):
    df, codes = panel(toy("S"))
    r = start_of(df, "S") + 80
    reverse_split(df, r, 30)
    df.loc[r + 5, "high"] = 6.0
    df.loc[r + 6:, ["open", "high", "low", "close", "unadj_close"]] = 3.0
    mkt = market(df)
    ev = rs.features(events(df, codes), df, pd.Series(1.0, index=mkt.index), market_features(df), mkt)
    ev = rs.entry_features(ev, df, 20, market_features(df))
    out = rs.audit(ev, df, mkt, 50.0, 2.0, {"bad prints (undone within 5 bars)": 0})
    assert len(out) == len(rs.AUDIT) * 3 and out["rules"].tolist()[:len(rs.AUDIT)] == ["0-20"] * len(rs.AUDIT)
    both = rs.combined(ev, df, mkt, 50.0, [2.0])
    run = out[out["audit"] == "as run"].set_index("rules")
    assert np.allclose(run["ret"], both.set_index("rules")["ret"]) and np.allclose(run["cagr_%"],
                                                                                  both.set_index("rules")["cagr_%"])
    text = capsys.readouterr().out
    assert "left out 0 split(s)" in text and "0-20: 0 trade(s) still open" in text
