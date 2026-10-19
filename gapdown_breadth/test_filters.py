"""Market-wide gap-downs, part 2: SPY's gap, the stock's filter values, the filter table's bar, the worst days,
and both scripts end to end on a made-up panel."""

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth_events as be  # noqa: E402
import filters as fl  # noqa: E402
from test_breadth_events import DATES, panel, toy  # noqa: E402


def test_spy_moves_take_the_gap_from_the_prior_close():
    market = pd.DataFrame({"date": DATES[[2, 0, 1]], "spy_open": [99.0, 100, 98], "spy_close": [101.0, 100, 100]})
    m = fl.spy_moves(market)
    assert list(m.index) == list(DATES[:3])
    assert np.isnan(m.loc[DATES[0], "spy_gap"]) and np.isclose(m.loc[DATES[1], "spy_gap"], -2.0)
    assert np.isclose(m.loc[DATES[2], "spy_gap"], -1.0) and np.isclose(m.loc[DATES[2], "spy_oc"], (101 / 99 - 1) * 100)
    assert fl.spy_moves(market.drop(columns=["spy_open"])).empty


def test_stock_features_are_read_at_the_prior_close():
    a = toy("AAA", [100, 104, 104, 101], opens=[100, 100, 104, 98.8])     # gaps -5% on day 3, after a flat day 2
    a["sma200"], a["sma50"], a["atr50"] = [90.0, 90, 110, 90], [120.0, 120, 100, 120], [2.0, 2, 2.6, 2]
    df, codes = panel(a)
    gap, _ = be.gaps(df, codes)
    t = pd.DataFrame({"row": [3], "date": [DATES[3]], "gap": [gap[3]]})
    moves = pd.DataFrame({"spy_gap": [-1.5], "spy_oc": [0.3]}, index=pd.DatetimeIndex([DATES[3]], name="date"))
    f = fl.stock_features(df, codes, t, moves).iloc[0]
    assert np.isclose(f["gap"], -5.0)
    assert np.isclose(f["vs200"], (104 / 110 - 1) * 100) and np.isclose(f["vs50"], (104 / 100 - 1) * 100)
    assert np.isclose(f["below_spy"], 3.5) and np.isclose(f["spy_gap"], -1.5)
    assert np.isclose(f["gap_atr"], 5.0 / (2.6 / 104 * 100))
    assert np.isclose(f["day_before"], 0.0)
    first = fl.stock_features(df, codes, pd.DataFrame({"row": [1], "date": [DATES[1]], "gap": [0.0]}), moves).iloc[0]
    assert np.isnan(first["day_before"]) and np.isnan(first["below_spy"])  # no day before day 0; no SPY that day


def test_buckets_leave_out_missing_values():
    t = pd.DataFrame({"vs200": [5.0, -1.0, np.nan], "vs50": [1.0, 0.0, 2.0], "below_spy": [0.5, 1.0, 3.0],
                      "gap_atr": [0.5, 2.0, np.nan], "day_before": [-2.5, -2.0, 2.0]})
    g = dict(fl.filter_groups(t))
    pick = lambda group: {lab: list(np.flatnonzero(m)) for lab, m in g[group]}
    assert pick("trend: vs its 200-day") == {"above": [0], "below": [1]}
    assert pick("trend: vs its 50-day") == {"above": [0, 2], "below": [1]}
    assert pick("gap below SPY's, points") == {"under 1": [0], "1 to 2.5": [1], "2.5+": [2]}
    assert pick("gap over its 50-day ATR") == {"under 1x": [0], "1x to 2x": [], "2x+": [1]}
    assert pick("the day before, %") == {"fell 2+": [0], "-2 to +2": [1], "rose 2+": [2]}
    assert list(g["every event"][0][1]) == [True, True, True]


def months_of_trades(start, end, ret, breadth=10.0):
    """Two trades on the first business day of each month from start to end, both with this ret."""
    d = pd.date_range(start, end, freq="BMS").repeat(2)
    return pd.DataFrame({"date": d, "ret": ret, "gross": ret, "vs_spy": ret, "breadth": breadth})


def test_the_bar_needs_t_above_2_and_every_block_from_2010():
    rnd = np.random.default_rng(0)
    good = months_of_trades("2010-01-01", "2026-06-30", 0.0)
    good["ret"] = 0.3 + rnd.normal(0, 0.2, len(good))
    gappy = good.copy()
    gappy.loc[pd.DatetimeIndex(gappy["date"]).year.isin([2016, 2017]), "ret"] = -1.0   # 2015-2019 loses
    old = months_of_trades("2005-01-01", "2009-12-31", -0.5, breadth=1.0)
    t = pd.concat([good, gappy, old], ignore_index=True)
    n, k = len(good), len(gappy)
    groups = [("g", [("good", np.r_[np.ones(n), np.zeros(k), np.ones(len(old))].astype(bool)),
                     ("gappy", np.r_[np.zeros(n), np.ones(k), np.zeros(len(old))].astype(bool))])]
    tab = fl.filter_table(t, groups, broad=5.0, split_year=2010).set_index("bucket")
    assert tab.loc["good", "passes"] and tab.loc["good", "t"] > 2 and tab.loc["good", "trades"] == n
    assert np.isclose(tab.loc["good", "before 2010"], -0.5) and tab.loc["good", "lone_trades"] == 0
    assert tab.loc["good", "broad_trades"] == n
    assert tab.loc["gappy", "ret"] > 0 and tab.loc["gappy", "2015-2019"] < 0 and not tab.loc["gappy", "passes"]
    assert list(tab.columns[-6:]) == ["2010-2014", "2015-2019", "2020-2024", "2025-2029", "before 2010", "passes"]


def test_day_totals_and_the_period_without_the_worst_and_best_days():
    t = pd.DataFrame({"date": pd.to_datetime(["2009-03-02", "2011-01-03", "2011-01-03", "2012-05-01", "2013-02-01"]),
                      "ret": [5.0, -2.0, -1.0, 4.0, 0.5], "breadth": [40.0, 25.0, 25.0, 6.0, 1.0]})
    moves = pd.DataFrame({"spy_gap": [-1.0], "spy_oc": [2.0]}, index=pd.DatetimeIndex(["2011-01-03"], name="date"))
    days = fl.day_totals(t, moves)
    assert days.loc["2011-01-03", "events"] == 2 and np.isclose(days.loc["2011-01-03", "total"], -3.0)
    assert np.isclose(days.loc["2011-01-03", "avg_ret"], -1.5) and np.isclose(days.loc["2011-01-03", "spy_oc"], 2.0)
    assert np.isnan(days.loc["2012-05-01", "spy_gap"])
    w = fl.without_extremes(days, n=1, split_year=2010).set_index("days")
    assert w.loc["every day", "events"] == 4 and np.isclose(w.loc["every day", "ret"], 1.5 / 4)
    assert np.isclose(w.loc["without the 1 worst days", "ret"], 4.5 / 2)
    assert np.isclose(w.loc["without the 1 best days", "ret"], -2.5 / 3)


def made_up_store(n_stocks=120, seed=1):
    """A panel of n_stocks S&P 500 members from mid-2008 to mid-2011 with the columns both scripts read: small
    daily gaps, a market-wide gap-down every 25th day and a few lone ones, plus SPY and the S&P's 200-day."""
    rnd = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-06-02", "2011-06-30")
    n = len(dates)
    wide = np.zeros(n, dtype=bool)
    wide[25::25] = True
    frames = []
    for i in range(n_stocks):
        g = rnd.normal(0, 0.006, n)
        g[wide] = rnd.normal(-0.035, 0.008, wide.sum())
        lone = rnd.random(n) < 0.01
        g[lone] = -rnd.uniform(0.025, 0.05, lone.sum())
        day = rnd.normal(0.0004, 0.012, n)
        c = 50 * np.cumprod((1 + g) * (1 + day))
        o = np.r_[50.0, c[:-1]] * (1 + g)
        hi, lo = np.maximum(o, c) * 1.005, np.minimum(o, c) * 0.995
        f = pd.DataFrame({"symbol": f"S{i:03d}", "date": dates, "open": o, "close": c, "bad_bar": 0, "bad_bar_dn": 0,
                          "in_sp500": 1, "in_sp400": 0, "in_sp600": 0, "listed": 1, "unadj_close": c, "dv_pct": 90.0,
                          "bars_since_break": np.nan, "tight_10": 0.05})
        s = pd.Series(c)
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.r_[np.nan, c[:-1]]), np.abs(lo - np.r_[np.nan, c[:-1]])))
        f["sma50"], f["sma200"] = s.rolling(50).mean().to_numpy(), s.rolling(200).mean().to_numpy()
        f["atr50"] = pd.Series(tr).rolling(50).mean().to_numpy()
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype("category")
    spy_c = 400 * np.cumprod(1 + rnd.normal(0.0003, 0.01, n))
    spy_o = np.r_[400.0, spy_c[:-1]] * np.where(wide, 0.98, 1.0)
    market = pd.DataFrame({"date": dates, "spy_open": spy_o, "spy_close": spy_c})
    spx = pd.DataFrame({"date": dates, "spx_above200": (np.arange(n) % 3 > 0).astype(int)})
    return df, market, spx


def run_main(monkeypatch, module, argv, store):
    df, market, spx = store
    logged = []

    def load_panel(cols, start=None, end=None, universe="dev", unlock_holdout=False):
        out = df[["symbol", "date"] + [c for c in cols if c not in ("symbol", "date")]]
        if start is not None:
            out = out[out["date"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["date"] <= pd.Timestamp(end)]
        return out.reset_index(drop=True)

    monkeypatch.setattr(be.datastore, "load_panel", load_panel)
    monkeypatch.setattr(be.datastore, "load_market", lambda cols=None: spx[["date"] + list(cols or [])])
    monkeypatch.setattr(be.pd, "read_parquet", lambda path, *a, **k: market)
    monkeypatch.setattr(be.research_log, "append", lambda **kw: logged.append(kw))
    monkeypatch.setattr(sys, "argv", [module.__name__ + ".py"] + argv)
    buf = io.StringIO()
    with redirect_stdout(buf):
        module.main()
    return buf.getvalue(), logged


def test_both_scripts_run_end_to_end(monkeypatch, tmp_path):
    store = made_up_store()
    out, logged = run_main(monkeypatch, be, ["--min-members", "50"], store)
    assert "THE EVENTS BY BREADTH" in out and "The pick, 2010 on:" in out, out
    assert logged and logged[0]["script"] == "breadth_events.py" and logged[0]["combinations"] == 2

    saved = tmp_path / "events.csv"
    out, logged = run_main(monkeypatch, fl, ["--min-members", "50", "--days", "5", "--save-trades", str(saved)], store)
    for part in ["BY YEAR, BEFORE 2010", "THE 5 WORST DAYS", "THE 5 BEST DAYS", "2010 ON, WITHOUT THE WORST",
                 "FILTERS ON THE STOCK", "The bar, set before the first run"]:
        assert part in out, (part, out)
    ev = pd.read_csv(saved)
    assert {"vs200", "vs50", "below_spy", "gap_atr", "day_before", "spy_gap", "breadth", "ret"} <= set(ev.columns)
    assert len(ev) > 100 and ev["gap"].between(-5, -2.5).all()
    assert logged[0]["script"] == "filters.py" and logged[0]["combinations"] == 13
    assert len(logged[0]["results"]["buckets"]) == 14
    worst = out.split("THE 5 WORST DAYS")[1].split("THE 5 BEST DAYS")[0]
    assert worst.count("\n20") == 5                                     # five dated rows
