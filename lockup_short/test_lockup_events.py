"""Lock-up expiry: which listings count, where day 0 falls, and the short's sums."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lockup_events as le  # noqa: E402


def toy(symbol, start, closes, unadj=None):
    n = len(closes)
    c = np.asarray(closes, float)
    return pd.DataFrame({"symbol": symbol, "date": pd.bdate_range(start, periods=n), "close": c,
                         "high": c, "unadj_close": c if unadj is None else np.asarray(unadj, float),
                         "bad_bar": 0, "bad_bar_dn": 0})


def panel(*frames):
    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    return df, df["symbol"].cat.codes.to_numpy()


def test_day0_is_first_bar_on_or_after_180_calendar_days():
    old = toy("OLD", "2010-01-01", np.full(400, 20.0))       # history starts with the data
    new = toy("NEW", "2011-03-01", np.full(200, 20.0))
    df, codes = panel(old, new)
    ev, left = le.find_events(df, codes, 180, pd.Timestamp("2010-01-01"))
    assert list(ev["symbol"]) == ["NEW"]
    assert left["history starts with the data"] == 1
    d0 = ev["date"].iloc[0]
    assert d0 >= pd.Timestamp("2011-03-01") + pd.Timedelta(days=180)
    assert d0 - pd.Timedelta(days=3) < pd.Timestamp("2011-03-01") + pd.Timedelta(days=180)


def test_relists_spacs_and_early_deaths_are_left_out():
    rel = toy("ABC~2", "2011-03-01", np.full(200, 20.0))
    spac = toy("SPC", "2011-03-01", np.r_[np.full(30, 10.02), np.full(170, 14.0)])
    dead = toy("DED", "2011-03-01", np.full(60, 20.0))
    keep = toy("IPO", "2011-03-01", np.full(200, 20.0))
    start = toy("OLD", "2010-01-01", np.full(10, 20.0))
    df, codes = panel(rel, spac, dead, keep, start)
    ev, left = le.find_events(df, codes, 180, pd.Timestamp("2010-01-01"))
    assert list(ev["symbol"]) == ["IPO"]
    assert left["returns to an exchange (~2)"] == 1
    assert left["SPAC-like start"] == 1
    assert left["stopped trading before day 0"] == 1
    ev, _ = le.find_events(df, codes, 180, pd.Timestamp("2010-01-01"), spacs="keep")
    assert sorted(ev["symbol"]) == ["IPO", "SPC"]
    assert ev.set_index("symbol").loc["SPC", "spac_like"]


def spx_flat(df):
    d = pd.DatetimeIndex(np.sort(df["date"].unique()))
    return pd.Series(np.full(len(d), 100.0), index=d)


def test_short_earns_the_fall_after_costs():
    c = np.full(200, 20.0)
    df, codes = panel(toy("IPO", "2011-03-01", c))
    ev, _ = le.find_events(df, codes, 180, pd.Timestamp("2010-01-01"))
    r = ev["row"].iloc[0]
    df.loc[r + 5:, "close"] = 18.0                          # falls 10% on day 5
    df["high"] = df["close"]
    out = le.window(ev, df, spx_flat(df), -5, 5, borrow_pct=5.0)
    cost = le.ce.round_trip_cost() + 5.0 * 10 / 252
    assert np.isclose(out["raw_-5:5"][0], -10.0)
    assert np.isclose(out["short_-5:5"][0], 10.0 - cost)
    assert np.isclose(out["hedged_-5:5"][0], 10.0 - cost)
    assert out["hit_-5:5"][0] == 0


def test_down_break_left_out_and_delisting_covers_at_last_close():
    df, codes = panel(toy("IPO", "2011-03-01", np.full(200, 20.0)))
    ev, _ = le.find_events(df, codes, 180, pd.Timestamp("2010-01-01"))
    r = ev["row"].iloc[0]
    df.loc[r + 2, "bad_bar_dn"] = 1
    out = le.window(ev, df, spx_flat(df), -5, 5, 5.0)
    assert np.isnan(out["short_-5:5"][0])
    out = le.window(ev, df, spx_flat(df), 3, 5, 5.0)         # after the break: fine
    assert np.isfinite(out["short_3:5"][0])

    df2, codes2 = panel(toy("IPO", "2011-03-01", np.full(200, 20.0)))
    ev2, _ = le.find_events(df2, codes2, 180, pd.Timestamp("2010-01-01"))
    r2 = ev2["row"].iloc[0]
    df2 = df2.iloc[:r2 + 3].copy()                            # stops trading on day 2
    df2.loc[r2 + 2, "close"] = 30.0
    df2["high"] = df2["close"]
    ev2["last"] = r2 + 2
    out = le.window(ev2, df2, spx_flat(df2), -5, 20, 5.0)
    assert np.isclose(out["raw_-5:20"][0], 50.0)
    assert out["hit_-5:20"][0] == 1


def test_parse_pick():
    assert le.parse_pick("-5:5") == (-5, 5)
    for bad in ["5:5", "3:-1", "x"]:
        try:
            le.parse_pick(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_volatility_picks_up_a_swing_on_day0():
    rng = np.random.default_rng(1)
    c = 20 * np.cumprod(1 + rng.normal(0, 0.01, 260))
    df, codes = panel(toy("IPO", "2011-03-01", c))
    df["open"], df["low"] = df["close"], df["close"]
    ev, _ = le.find_events(df, codes, 180, pd.Timestamp("2010-01-01"))
    r = ev["row"].iloc[0]
    df.loc[r:, "close"] *= 1.2                                 # +20% on day 0 only
    df["high"], df["open"], df["low"] = df["close"], df["close"], df["close"]
    daily, blocks = le.volatility(ev, df, spx_flat(df))
    d = daily.set_index("day")
    assert d.loc[0, "move_x"] > 10 and d.loc[1, "move_x"] < 5
    assert np.isclose(d.loc[0, "abs_move"], 20.0, atol=3.0)
    b = blocks.set_index("days")
    assert b.loc["0 to 4", "abs_5d"] > 15 and b.loc["5 to 9", "abs_5d"] < 10


def test_bins_label_and_split_every_value_once():
    v = pd.Series([0.5, 2.0, 7.0, 60.0, np.nan])
    b = le.bins_of(v, le.PRICE_BINS, lambda x: f"${x:g}")
    labels = [lab for lab, _ in b]
    assert labels[0] == "under $2" and labels[-1] == "$50+" and "$5 to $10" in labels
    hits = np.sum([m for _, m in b], axis=0)
    assert list(hits) == [1, 1, 1, 1, 0]
