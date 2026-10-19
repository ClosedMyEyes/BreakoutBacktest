"""Market-wide gap-downs: the gap, the day's breadth, which gaps count, and the day trade's sums."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth_events as be  # noqa: E402
import rules  # noqa: E402  (breadth_events put core/ on the path)

DATES = pd.bdate_range("2012-01-02", periods=6)


def toy(symbol, closes, opens=None, member=1):
    """One stock over DATES: closes, opens (default: the prior close, no gap), an S&P 500 member
    throughout (member=1) or not, passing the universe on every day."""
    c = np.asarray(closes, dtype=np.float64)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, dtype=np.float64)
    n = len(c)
    return pd.DataFrame({"symbol": symbol, "date": DATES[:n], "open": o, "close": c, "bad_bar": 0, "bad_bar_dn": 0,
                         "in_sp500": np.full(n, member), "in_sp400": 0, "in_sp600": 0, "listed": 1,
                         "unadj_close": c, "dv_pct": 90.0, "bars_since_break": np.nan, "tight_10": 0.05})


def panel(*frames):
    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    return df, df["symbol"].cat.codes.to_numpy()


def row_of(df, sym, day):
    return int(df.index[(df["symbol"] == sym) & (df["date"] == DATES[day])][0])


def test_gaps_use_the_same_stocks_prior_close_and_flag_price_breaks():
    a = toy("AAA", [20, 20, 20, 20], opens=[20, 19, 21, 20])
    b = toy("BBB", [50, 50, 50, 50], opens=[40, 50, 50, 55])
    b.loc[3, "bad_bar"] = 1
    df, codes = panel(a, b)
    gap, brk = be.gaps(df, codes)
    assert np.isnan(gap[0]) and np.isnan(gap[4])                     # each stock's first row
    assert np.allclose(gap[1:4], [-5.0, 5.0, 0.0])
    assert np.isclose(gap[7], 10.0) and brk[7] and not brk[:7].any()


def test_breadth_counts_members_at_the_prior_close_and_skips_breaks_and_thin_days():
    frames = [toy(f"S{i}", [10] * 4, opens=[10, 9.7 if i < 2 else 10, 10, 9.0]) for i in range(10)]
    joiner = toy("NEW", [10] * 4, opens=[10, 9.0, 10, 10], member=0)
    joiner.loc[1:, "in_sp500"] = 1                                   # joins on day 1: not a member at day 0's close
    leaver = toy("OLD", [10] * 4, opens=[10, 9.0, 10, 10])
    leaver.loc[1:, "in_sp500"] = 0                                   # leaves on day 1: was a member at day 0's close
    broken = toy("BRK", [10] * 4, opens=[10, 5.0, 10, 10])
    broken.loc[1, "bad_bar_dn"] = 1                                  # a bad split adjustment, not a gap
    df, codes = panel(*frames, joiner, leaver, broken)
    gap, brk = be.gaps(df, codes)
    br = be.breadth(df, codes, gap, brk, thr=2.5, min_members=5)
    d1, d3 = DATES[1], DATES[3]
    assert br.loc[d1, "members"] == 11                               # 10 + OLD; NEW and BRK left out
    assert br.loc[d1, "down"] == 3                                   # S0, S1 (-3%) and OLD (-10%)
    assert np.isclose(br.loc[d1, "pct"], 3 / 11 * 100)
    assert br.loc[d3, "down"] == 10 and br.loc[d3, "members"] == 12  # NEW counts from day 2's close
    assert DATES[0] not in br.index                                  # no prior close on day 0
    thin = be.breadth(df, codes, gap, brk, thr=2.5, min_members=12)
    assert list(thin.index) == [DATES[2], d3]                        # day 1 had 11


def test_bucket_edges():
    got = be.bucket([0.0, 4.99, 5.0, 9.99, 10.0, 25.0, 30.0, 100.0, np.nan])
    assert list(got) == ["under 5%", "under 5%", "5-10%", "5-10%", "10-20%", "20-30%", "30%+", "30%+", None]


def test_events_take_the_band_the_universe_at_the_prior_close_and_no_break_days():
    rows_in = toy("INN", [100] * 6, opens=[100, 97, 100, 95, 100, 100])     # -3% counts, -5% is out of the band
    small = toy("SML", [100] * 6, opens=[100, 98, 100, 100, 100, 100])      # -2%: too small
    joined = toy("JND", [100] * 6, opens=[100, 97, 100, 100, 100, 100], member=0)
    joined.loc[1:, "in_sp500"] = 1                                           # a member only from the gap day
    cheap = toy("CHP", [100] * 6, opens=[100, 100, 100, 100, 96, 100])
    cheap.loc[3, "unadj_close"] = 4.0                                        # under $5 at the prior close
    broke = toy("BRK", [100] * 6, opens=[100, 100, 97, 100, 100, 100])
    broke.loc[2, "bad_bar_dn"] = 1
    df, codes = panel(rows_in, small, joined, cheap, broke)
    gap, brk = be.gaps(df, codes)
    P = dict(rules.UNIVERSE_DEFAULTS, size="sp500")
    got = be.events(df, codes, gap, brk, P, 2.5, 5.0)
    assert list(got) == [row_of(df, "INN", 1)]
    wide = be.events(df, codes, gap, brk, dict(P, size="all"), 2.5, np.inf)
    assert sorted(wide) == sorted([row_of(df, "INN", 1), row_of(df, "INN", 3), row_of(df, "JND", 1)])
    late = be.events(df, codes, gap, brk, P, 2.5, 5.0, start=str(DATES[2].date()))
    assert len(late) == 0


def test_day_trades_and_spy_after_costs():
    a = toy("AAA", [100, 100, 99, 100], opens=[100, 100, 97, 100])         # opens -3%, closes +2.06% from the open
    df, codes = panel(a)
    gap, _ = be.gaps(df, codes)
    br = pd.DataFrame({"members": [200, 200, 200], "down": [2, 20, 1], "pct": [1.0, 10.0, 0.5]}, index=DATES[1:4])
    market = pd.DataFrame({"date": DATES[:4], "spy_open": [400.0, 400, 400, 400], "spy_close": [400.0, 400, 404, 400]})
    spy = be.spy_days(market, br, cost=0.02)
    assert list(spy["date"]) == list(DATES[1:4])
    assert np.isclose(spy.loc[1, "gross"], 1.0) and np.isclose(spy.loc[1, "ret"], 0.98)
    assert list(spy["bin"]) == ["under 5%", "10-20%", "under 5%"]
    t = be.day_trades(df, np.array([2]), gap, br, spy, cost=0.10)
    gross = (99 / 97 - 1) * 100
    assert np.isclose(t.loc[0, "gross"], gross) and np.isclose(t.loc[0, "ret"], gross - 0.10)
    assert np.isclose(t.loc[0, "vs_spy"], gross - 1.0)
    assert np.isclose(t.loc[0, "gap"], -3.0) and t.loc[0, "bin"] == "10-20%" and t.loc[0, "symbol"] == "AAA"
    none = be.day_trades(df, np.array([0]), gap, br, spy, cost=0.10)        # day 0 has no breadth reading
    assert none.empty
    assert be.spy_days(market.drop(columns=["spy_open"]), br, 0.02).empty


def test_tables_split_broad_from_lone_and_by_period():
    t = pd.DataFrame({"date": pd.to_datetime(["2009-03-02", "2012-03-02", "2012-03-02", "2013-05-01"]),
                      "gross": [1.0, 2.0, 0.0, -1.0], "breadth": [12.0, 8.0, 8.0, 1.0]})
    t["ret"], t["vs_spy"], t["bin"] = t["gross"] - 0.1, t["gross"] - 0.5, be.bucket(t["breadth"])
    tb = be.by_bin(t, be.stats, broad=5.0, split_year=2010).set_index(["breadth", "years"])
    assert tb.loc[("5-10%", "2010 on"), "trades"] == 2 and np.isclose(tb.loc[("5-10%", "2010 on"), "ret"], 0.9)
    assert tb.loc[("broad (5%+)", "before 2010"), "trades"] == 1
    assert tb.loc[("under 5%", "2010 on"), "trades"] == 1
    yr = be.by_period(t[t["breadth"] >= 5], t[t["breadth"] < 5], t.iloc[:0], be.year).set_index("period")
    assert yr.loc[2012, "broad_trades"] == 2 and yr.loc[2013, "lone_trades"] == 1
    assert np.isclose(yr.loc[2012, "broad_ret"], 0.9) and np.isnan(yr.loc[2012, "spy_broad_ret"])
    assert list(be.block(pd.to_datetime(["2009-01-02", "2010-01-04"]))) == ["2005-2009", "2010-2014"]
