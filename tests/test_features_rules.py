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
