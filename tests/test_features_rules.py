"""No look-ahead, template rules, labels and the Holdout lock."""

import numpy as np
import pandas as pd
import pytest

import build_features as bf
import datastore
import rules
from data_sources import SyntheticSource


def two_symbols():
    src = SyntheticSource(n_symbols=4, start_date="2010-01-01", end_date="2013-12-31")
    frames = [src.symbol_prices(s) for s in src.all_symbols()[:2]]
    p = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    return p


def test_features_use_no_future_bars():
    p = two_symbols()
    f1 = bf.symbol_features(p)
    cut = p.groupby("symbol").cumcount() == 400           # change everything after bar 400
    after = p.groupby("symbol").cumcount() > 400
    p2 = p.copy()
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        p2.loc[after, c] = p2.loc[after, c] * 3.7
    f2 = bf.symbol_features(p2)
    upto = ~after
    for col in f1.columns:
        a, b = f1.loc[upto, col].to_numpy(np.float64), f2.loc[upto, col].to_numpy(np.float64)
        assert np.allclose(a, b, equal_nan=True), col
    assert cut.any()


def test_symbols_do_not_leak_into_each_other():
    p = two_symbols()
    f = bf.symbol_features(p)
    second = p["symbol"] == p["symbol"].unique()[1]
    first_rows = f[second].iloc[:49]
    assert first_rows["sma50"].isna().all()
    assert np.isfinite(f[second]["sma50"].iloc[49])


def test_forward_labels():
    p = two_symbols()
    lab = bf.symbol_labels(p, delisted_set=set())
    s = p[p["symbol"] == p["symbol"].unique()[0]].reset_index(drop=True)
    l0 = lab.loc[s.index]
    i = 100
    expected = s["close"].iloc[i + 20] / s["open"].iloc[i + 1] - 1
    assert l0["fwd20"].iloc[i] == pytest.approx(expected, rel=1e-5)
    assert np.isnan(l0["fwd20"].iloc[-5])                      # active stock near data end


def test_delisted_labels_use_last_price():
    p = two_symbols()
    sym = p["symbol"].unique()[0]
    lab = bf.symbol_labels(p, delisted_set={sym})
    s = p[p["symbol"] == sym].reset_index(drop=True)
    i = len(s) - 5
    expected = s["close"].iloc[-1] / s["open"].iloc[i + 1] - 1
    assert lab.loc[i, "fwd20"] == pytest.approx(expected, rel=1e-5)


def template_row(**over):
    row = {"close": 100, "sma50": 95, "sma150": 90, "sma200": 85, "sma200_up21": 1,
           "hi52_c": 110, "lo52_c": 60, "rs_rank": 85}
    row.update(over)
    return pd.DataFrame([row])


@pytest.mark.parametrize("change,passes", [
    ({}, True),
    ({"close": 89}, False),               # rule 1
    ({"sma150": 80}, False),              # rule 2 (150 below 200)
    ({"sma200_up21": 0}, False),          # rule 3
    ({"sma50": 88}, False),               # rule 4
    ({"sma50": 101}, False),              # rule 5
    ({"lo52_c": 80}, False),              # rule 6: only 25% above low
    ({"hi52_c": 140}, False),             # rule 7: 28.6% below high
    ({"rs_rank": 69}, False),             # rule 8
])
def test_template_rules(change, passes):
    assert rules.template_mask(template_row(**change), {})[0] == passes


def test_template_knobs():
    df = template_row(hi52_c=125)                                  # 20% below high
    assert rules.template_mask(df, {"max_below_high": 25})[0]
    assert not rules.template_mask(df, {"max_below_high": 15})[0]


def test_holdout_is_locked():
    with pytest.raises(datastore.HoldoutLocked):
        datastore.check_universe("holdout", unlock_holdout=False)
    with pytest.raises(datastore.HoldoutLocked):
        datastore.check_universe("all", unlock_holdout=False)
    datastore.check_universe("holdout", unlock_holdout=True)
    datastore.check_universe("dev", unlock_holdout=False)


def test_universe_needs_major_exchange_listing():
    df = pd.DataFrame({"unadj_close": [20.0, 20.0, 3.0], "dv_pct": [90.0, 90.0, 90.0],
                       "listed": [1, 0, 1], "bars_since_break": [np.nan] * 3, "tight_10": [0.05] * 3})
    assert rules.universe_mask(df, {}).tolist() == [True, False, False]


def test_universe_skips_stocks_pinned_at_a_buyout_price():
    df = pd.DataFrame({"unadj_close": 20.0, "dv_pct": 90.0, "listed": 1, "bars_since_break": np.nan,
                       "tight_10": [0.05, 0.015, np.nan]})     # 5% range, 1.5% range, too new to tell
    assert rules.universe_mask(df, {}).tolist() == [True, False, True]
    assert rules.universe_mask(df, {"min_range10": 0}).all()


def one_stock(close, unadj):
    n = len(close)
    close, unadj = np.asarray(close, float), np.asarray(unadj, float)
    return pd.DataFrame({"symbol": "AAA", "date": pd.bdate_range("2020-01-01", periods=n),
                         "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1e6, "unadj_close": unadj, "turnover": unadj * 1e6})


def test_clean_split_is_not_a_break():
    close = np.full(300, 50.0)
    unadj = np.where(np.arange(300) < 150, 100.0, 50.0)       # 2-for-1 split, adjusted smooth
    f = bf.symbol_features(one_stock(close, unadj))
    assert f["bars_since_break"].isna().all()


def test_price_break_blocks_new_entries():
    i = np.arange(300)
    close = np.where(i < 150, 2.0, 16.0)                      # adjusted close jumps 8x...
    unadj = np.where(i < 150, 0.2, 16.0)                      # ...on a day the factor moves 10x
    f = bf.symbol_features(one_stock(close, unadj))
    s = f["bars_since_break"]
    assert s.iloc[:150].isna().all() and s.iloc[150] == 0 and s.iloc[160] == 10
    df = pd.DataFrame({"unadj_close": unadj, "dv_pct": 90.0, "listed": 1, "bars_since_break": s, "tight_10": 0.05})
    m = rules.universe_mask(df, {"min_price": 0})
    assert m[:150].all() and not m[150:].any()                # blocked for 252 bars from the break
    assert rules.universe_mask(df, {"min_price": 0, "block_after_break": 0}).all()


def test_extreme_bar_is_a_break_without_adjustment():
    close = np.where(np.arange(100) < 50, 1.0, 12.0)          # 12x with no factor change
    f = bf.symbol_features(one_stock(close, close))
    assert f["bars_since_break"].iloc[50] == 0


def test_upward_break_is_a_bad_bar_downward_is_not():
    i = np.arange(200)
    close = np.where(i < 100, 10.0, np.where(i < 150, 1000.0, 50.0))   # 100x up, then -95%
    f = bf.symbol_features(one_stock(close, close))
    assert f["bars_since_break"].iloc[100] == 0 and f["bars_since_break"].iloc[150] == 0
    assert np.flatnonzero(f["bad_bar"].to_numpy()).tolist() == [100]


def test_forward_returns_spanning_an_upward_break_are_left_out():
    i = np.arange(200)
    close = np.where(i < 100, 10.0, np.where(i < 150, 1000.0, 50.0))
    lab = bf.symbol_labels(one_stock(close, close), delisted_set=set())
    assert lab["fwd20"].iloc[:80].notna().all()                # window ends before the jump
    assert lab["fwd20"].iloc[80:100].isna().all()              # window holds the 100x bar
    assert lab["fwdmax60"].iloc[60:100].isna().all()
    assert lab["fwd5"].iloc[148] == pytest.approx(-0.95)       # the crash counts


def test_size_other_means_in_no_sp_index():
    df = pd.DataFrame({"unadj_close": 20.0, "dv_pct": 90.0, "listed": 1, "bars_since_break": np.nan,
                       "tight_10": 0.05, "in_sp500": [1, 0, 0], "in_sp400": [0, 1, 0], "in_sp600": [0, 0, 0]})
    assert rules.universe_mask(df, {"size": "other"}).tolist() == [False, False, True]
    assert rules.universe_mask(df, {"size": "sp400_600"}).tolist() == [False, True, False]
