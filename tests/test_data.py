"""Listing-run segmentation (build_data) and the company-level Dev/Holdout split."""

import numpy as np
import pandas as pd

import config
import datastore
from data_sources import listed_segments
from make_split import make_split


def history(listed):
    n = len(listed)
    return pd.DataFrame({"symbol": "AAA", "date": pd.bdate_range("2020-01-01", periods=n),
                         "close": np.arange(1.0, n + 1), "listed": listed})


def test_otc_tail_and_head_are_dropped():
    segs = listed_segments(history([0] * 5 + [1] * 30 + [0] * 40), max_gap=20)
    assert len(segs) == 1
    assert segs[0]["symbol"].unique().tolist() == ["AAA"]
    assert segs[0]["close"].tolist() == list(np.arange(6.0, 36.0))     # only the listed bars


def test_short_otc_gap_stays_inside_the_run():
    segs = listed_segments(history([1] * 30 + [0] * 19 + [1] * 30), max_gap=20)
    assert len(segs) == 1 and len(segs[0]) == 79
    assert (segs[0]["listed"] == 0).sum() == 19                         # kept, but no entries there


def test_return_to_an_exchange_is_a_new_symbol():
    segs = listed_segments(history([1] * 30 + [0] * 20 + [1] * 30 + [0] * 50 + [1] * 10), max_gap=20)
    assert [s["symbol"].iloc[0] for s in segs] == ["AAA", "AAA~2", "AAA~3"]
    assert [len(s) for s in segs] == [30, 30, 10]
    assert segs[1]["close"].iloc[0] == 51.0
    assert (pd.concat(segs)["listed"] == 1).all()


def test_never_listed_gives_nothing():
    assert listed_segments(history([0] * 30)) == []


def test_split_keeps_a_company_on_one_side():
    syms = [f"S{i}" for i in range(200)]
    df = pd.DataFrame({"symbol": syms + ["S7~2", "S7~3", "S8~2"], "sector": ["Tech"] * 203})
    out = make_split(df, seed=1).set_index("symbol")["split"]
    assert out["S7~2"] == out["S7~3"] == out["S7"]
    assert out["S8~2"] == out["S8"]
    assert abs((out[syms] == "dev").sum() - 100) <= 1                   # halves count companies


def test_relisted_symbol_takes_its_side_from_a_locked_split(tmp_path, monkeypatch):
    split, syms = tmp_path / "split.csv", tmp_path / "symbols.parquet"
    pd.DataFrame({"symbol": ["AAA", "BBB"], "sector": ["Tech"] * 2, "split": ["dev", "holdout"]}).to_csv(split, index=False)
    pd.DataFrame({"symbol": ["AAA", "AAA~2", "BBB", "BBB~2", "NEW"]}).to_parquet(syms)
    monkeypatch.setattr(config, "SPLIT_FILE", str(split))
    monkeypatch.setattr(config, "SYMBOLS_FILE", str(syms))
    assert datastore.tradable_symbols("dev") == {"AAA", "AAA~2"}
    assert datastore.tradable_symbols("holdout") == {"BBB", "BBB~2"}
