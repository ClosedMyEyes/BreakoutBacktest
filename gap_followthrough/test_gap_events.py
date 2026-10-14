"""Big gaps: which days count, and the trade's sums in each direction."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_events as ge  # noqa: E402


def toy(symbol, n=120, price=20.0, vol=1000.0):
    c = np.full(n, price)
    return pd.DataFrame({"symbol": symbol, "date": pd.bdate_range("2012-01-02", periods=n), "open": c.copy(),
                         "high": c.copy(), "low": c.copy(), "close": c.copy(), "volume": np.full(n, vol),
                         "bad_bar": 0, "bad_bar_dn": 0})


def panel(*frames):
    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    return df, df["symbol"].cat.codes.to_numpy()


def gap_at(df, row, pct, vol_x=4.0):
    """Open row `row` pct% from the prior close and keep that price after, on vol_x volume."""
    px = df.loc[row - 1, "close"] * (1 + pct / 100)
    sym = df.loc[row, "symbol"]
    later = (df.index >= row) & (df["symbol"] == sym)
    df.loc[later, ["open", "high", "low", "close"]] = px
    df.loc[row, "volume"] = df.loc[row - 1, "volume"] * vol_x


def test_stock_bounds_and_usual_volume_stay_inside_each_stock():
    df, codes = panel(toy("AAA", n=30, vol=100.0), toy("BBB", n=30, vol=500.0))
    first, last = ge.stock_bounds(codes)
    assert first[0] == 0 and last[0] == 29 and first[30] == 30 and last[59] == 59
    u = ge.usual_volume(df["volume"].to_numpy(), first)
    assert np.isnan(u[19]) and np.isclose(u[20], 100.0)            # 20 bars before row 20
    assert np.isnan(u[45]) and np.isclose(u[50], 500.0)            # BBB starts fresh at row 30


def test_find_gaps_needs_size_and_volume_and_keeps_the_first_in_20_bars():
    df, codes = panel(toy("BIG"), toy("SML"), toy("THN"), toy("TWO"))
    rows = {s: int(df.index[df["symbol"] == s][0]) for s in ["BIG", "SML", "THN", "TWO"]}
    gap_at(df, rows["BIG"] + 60, 8)
    gap_at(df, rows["SML"] + 60, 3)                                 # too small
    gap_at(df, rows["THN"] + 60, 8, vol_x=2)                        # too little volume
    gap_at(df, rows["TWO"] + 60, -8)
    gap_at(df, rows["TWO"] + 70, 8)                                 # 10 bars later: a repeat
    ev, counts = ge.find_gaps(df, codes)
    got = dict(zip(df["symbol"].astype(str).to_numpy()[ev["row"]], ev["sign"]))
    assert got == {"BIG": 1, "TWO": -1}
    assert counts["repeat within the dedupe bars"] == 1
    assert np.isclose(ev.loc[ev["sign"] > 0, "gap"].iloc[0], 8.0)


def test_a_gap_day_with_a_price_break_is_dropped():
    a, b = toy("AAA", price=20.0), toy("BBB", price=40.0)
    b["volume"] = 1000.0
    df, codes = panel(a, b)
    r = int(df.index[df["symbol"] == "AAA"][0]) + 60
    gap_at(df, r, 10)
    df.loc[r, "bad_bar"] = 1                                        # a split, not news
    ev, counts = ge.find_gaps(df, codes)
    assert ev.empty and counts["price break on the day"] == 1


def spx_flat(df):
    d = pd.DatetimeIndex(np.sort(df["date"].unique()))
    return pd.Series(np.full(len(d), 100.0), index=d)


def test_trades_go_with_the_gap_after_costs():
    df, codes = panel(toy("UPP"), toy("DWN"))
    ru = int(df.index[df["symbol"] == "UPP"][0]) + 60
    rd = int(df.index[df["symbol"] == "DWN"][0]) + 60
    gap_at(df, ru, 10)                                              # 20 -> 22
    gap_at(df, rd, -10)                                             # 20 -> 18
    df.loc[ru + 1, "open"] = 22.5
    df.loc[ru + 5:ru + 59, ["close", "high", "low", "open"]] = 24.2  # +10% from the day 0 close
    df.loc[rd + 5:rd + 59, ["close", "high", "low", "open"]] = 16.2  # -10% from the day 0 close
    ev, _ = ge.find_gaps(df, codes)
    ev = ev.sort_values("sign").reset_index(drop=True)              # DWN, UPP
    ev["date"] = df["date"].to_numpy()[ev["row"]]
    out = ge.trades(ev, df, spx_flat(df), "close", 10, borrow_pct=5.0)
    cost = ge.ce.round_trip_cost()
    assert np.isclose(out["ret_close:10"][1], 10.0 - cost)
    assert np.isclose(out["ret_close:10"][0], 10.0 - cost - 5.0 * 10 / 252)
    assert np.isclose(out["hedged_close:10"][1], 10.0 - cost)
    op = ge.trades(ev, df, spx_flat(df), "open", 10, borrow_pct=5.0)
    assert np.isclose(op["ret_open:10"][1], (24.2 / 22.5 - 1) * 100 - cost)
    assert list(out["adv_close:10"]) == [0.0, 0.0]


def test_breaks_in_the_trades_favour_are_left_out_and_adverse_moves_counted():
    df, codes = panel(toy("UPP"), toy("DWN"))
    ru = int(df.index[df["symbol"] == "UPP"][0]) + 60
    rd = int(df.index[df["symbol"] == "DWN"][0]) + 60
    gap_at(df, ru, 10)
    gap_at(df, rd, -10)
    df.loc[ru + 3, "bad_bar"] = 1                                   # an untrusted jump up: no gain for a buy
    df.loc[rd + 2, "high"] = 18 * 1.25                              # the short goes 25% against
    ev, _ = ge.find_gaps(df, codes)
    ev = ev.sort_values("sign").reset_index(drop=True)
    out = ge.trades(ev, df, spx_flat(df), "close", 10, borrow_pct=0.0)
    assert np.isnan(out["ret_close:10"][1])
    assert out["adv_close:10"][0] == 1.0


def test_parse_pick_and_bins():
    assert ge.parse_pick("close:20") == ("close", 20)
    assert ge.parse_pick("open:5") == ("open", 5)
    for bad in ["noon:5", "close:0"]:
        try:
            ge.parse_pick(bad)
            assert False, bad
        except ValueError:
            pass
    v = pd.Series([5.0, 9.9, 10.0, 25.0])
    b = ge.bins_of(v, ge.edges_of(5.0, ge.GAP_BINS), lambda x: f"{x:g}%")
    assert [lbl for lbl, _ in b] == ["5% to 10%", "10% to 20%", "20%+"]
    assert [int(m.sum()) for _, m in b] == [2, 1, 1]
    p = ge.bins_of(pd.Series([3.0, 60.0]), ge.PRICE_BINS, lambda x: f"${x:g}")
    assert p[0][0] == "under $10" and p[-1][0] == "$50+"


def test_the_day_0_open_entry_catches_the_day_after_the_open():
    df, codes = panel(toy("UPP"), toy("DWN"))
    ru = int(df.index[df["symbol"] == "UPP"][0]) + 60
    rd = int(df.index[df["symbol"] == "DWN"][0]) + 60
    gap_at(df, ru, 10)                                              # opens 22
    gap_at(df, rd, -10)                                             # opens 18
    df.loc[ru, "close"] = 23.1                                      # +5% from the open
    df.loc[rd, "close"] = 17.1                                      # -5% from the open
    ev, _ = ge.find_gaps(df, codes)
    ev = ev.sort_values("sign").reset_index(drop=True)              # DWN, UPP
    out = ge.trades(ev, df, spx_flat(df), "day0", 0, borrow_pct=5.0)
    cost = ge.ce.round_trip_cost()
    assert np.isclose(out["ret_day0:0"][1], 5.0 - cost)
    assert np.isclose(out["ret_day0:0"][0], 5.0 - cost - 5.0 / 252)
    assert np.isclose(out["hedged_day0:0"][1], 5.0 - cost)
    assert ge.parse_pick("day0:0") == ("day0", 0)


def with_feature_columns(df):
    c = df["close"].to_numpy()
    return df.assign(unadj_close=c, dv50=c * df["volume"], sma200=np.nan, in_sp500=0, in_sp400=0, in_sp600=0,
                     hi52_c=25.0, lo52_c=20.0)


def test_the_lead_up_is_measured_in_the_gaps_direction_up_to_the_prior_close():
    df, codes = panel(toy("UPP"), toy("DWN"))
    rows = {}
    for s, pct in [("UPP", 8), ("DWN", -8)]:
        r = int(df.index[df["symbol"] == s][0]) + 60
        df.loc[r - 10:r - 1, "close"] = 22.0                        # +10% over the last 10 bars
        df.loc[r - 10:r - 1, "volume"] = 2000.0                     # twice the usual volume
        df.loc[r - 5, "high"], df.loc[r - 15, "low"] = 25.0, 18.0   # a 7-point range in 20 bars
        gap_at(df, r, pct)
        rows[s] = r
    df = with_feature_columns(df)
    ev, _ = ge.find_gaps(df, codes)
    ev = ev.sort_values("sign").reset_index(drop=True)              # DWN, UPP
    first, _ = ge.stock_bounds(codes)
    spx200 = pd.Series(1.0, index=pd.DatetimeIndex(np.sort(df["date"].unique())))
    ev = ge.features(ev, df, spx200, ge.usual_volume(df["volume"].to_numpy(), first))
    assert np.allclose(ev["move10"], [-10.0, 10.0]) and np.allclose(ev["move20"], [-10.0, 10.0])
    assert np.allclose(ev["range20"], 7 / 22 * 100)
    assert np.allclose(ev["build10"], 2.0)
    assert np.allclose(ev["from_extreme"], [10.0, 12.0])            # 22 vs a 20 low, 22 vs a 25 high
    assert np.isnan(ev["above200"]).all() and (ev["spx_above200"] == 1).all()


def test_keep_only_keeps_one_bucket():
    ev = pd.DataFrame({"move20": [-15.0, 0.0, 5.0, 10.0, np.nan], "build10": [1.0, 2.0, 0.5, 1.0, 1.0]})
    assert list(ge.keep_only(ev, ["move20:0:10"])["move20"]) == [0.0, 5.0]
    assert list(ge.keep_only(ev, ["move20:-inf:10", "build10:0:1.5"])["move20"]) == [-15.0, 5.0]
