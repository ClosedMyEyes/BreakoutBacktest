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
