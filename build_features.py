"""
Phase 2 — Feature build
=======================
Computes every indicator once so grid runs only read results. Slow step; run
once per data refresh.

Writes four row-aligned or per-day files to config.DATA_DIR:
  features.parquet         per-symbol indicators (same row order as prices.parquet)
  ranks.parquet            cross-sectional RS and liquidity ranks (same row order)
  labels.parquet           forward returns for Mode A ONLY (same row order).
                           Kept in a separate file so no rule can read the future
                           by accident (ground rule 3.1).
  market_features.parquet  index trend, distribution days, breadth (one row per day)

No look-ahead: every feature on day t uses bars up to and including t. Pivots
(e.g. pivot_50) are the high of the last N bars through t, i.e. the buy-stop
level for day t+1.

HOW TO RUN:
  python build_features.py
  python build_features.py --chunk 1500     # fewer symbols per chunk if RAM is tight
"""

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, default=3000, help="Symbols per chunk")
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

F32 = np.float32
PIVOT_LOOKBACKS = sorted(set(config.BREAKOUT_LOOKBACKS + config.TIGHTNESS_WINDOWS))


# =============================================================================
# GROUP-AWARE HELPERS
# Rows are sorted by (symbol, date). A plain rolling/shift over the whole column
# is correct inside each symbol; the first rows of each symbol would borrow the
# previous symbol's data, so they are masked using `pos` (bar number within
# symbol) and `rpos` (bars remaining until the symbol's last row).
# =============================================================================

def g_shift(s, k, pos, rpos):
    out = s.shift(k)
    if k > 0:
        out[pos < k] = np.nan
    elif k < 0:
        out[rpos < -k] = np.nan
    return out


def g_roll(s, w, how, pos):
    r = getattr(s.rolling(w, min_periods=w), how)()
    r[pos < w - 1] = np.nan
    return r


def g_roll_fwd(s, w, how, sym, pos, rpos):
    """Rolling over the NEXT w bars (t+1 .. t+w), truncated at the symbol's last
    row (delisted stocks stop at their last price). Labels only."""
    r = s.groupby(sym, observed=True).transform(
        lambda x: getattr(x.iloc[::-1].rolling(w, min_periods=1), how)().iloc[::-1])
    return g_shift(r, -1, pos, rpos)


def ema(s, span, sym, pos):
    out = s.groupby(sym, observed=True).transform(lambda x: x.ewm(span=span, adjust=False).mean())
    out[pos < span - 1] = np.nan
    return out


# =============================================================================
# PER-SYMBOL FEATURES (one chunk of whole symbols at a time)
# =============================================================================

def symbol_features(p):
    pos = p.groupby("symbol", observed=True).cumcount()
    rpos = p.groupby("symbol", observed=True).cumcount(ascending=False)
    c, h, l, v = p["close"], p["high"], p["low"], p["volume"]
    f = pd.DataFrame(index=p.index)

    # Moving averages
    for n in [10, 50, 150, 200]:
        f[f"sma{n}"] = g_roll(c, n, "mean", pos)
    f["ema21"] = ema(c, 21, p["symbol"], pos)

    # 200-day trend tests (template rule 3 variants)
    for n in [21, 63, 105]:
        f[f"sma200_up{n}"] = (f["sma200"] > g_shift(f["sma200"], n, pos, rpos)).astype("int8")
    up1 = (f["sma200"] > g_shift(f["sma200"], 1, pos, rpos))
    run_id = (~up1).cumsum()
    f["sma200_rising_days"] = up1.groupby(run_id).cumsum().astype("int16")

    # 52-week high / low, closing and intraday
    f["hi52_c"] = g_roll(c, 252, "max", pos)
    f["lo52_c"] = g_roll(c, 252, "min", pos)
    f["hi52_i"] = g_roll(h, 252, "max", pos)
    f["lo52_i"] = g_roll(l, 252, "min", pos)

    # RS scores (ranked cross-sectionally later)
    r63  = c / g_shift(c, 63, pos, rpos) - 1
    r126 = c / g_shift(c, 126, pos, rpos) - 1
    r189 = c / g_shift(c, 189, pos, rpos) - 1
    r252 = c / g_shift(c, 252, pos, rpos) - 1
    q2 = g_shift(c, 63, pos, rpos) / g_shift(c, 126, pos, rpos) - 1
    q3 = g_shift(c, 126, pos, rpos) / g_shift(c, 189, pos, rpos) - 1
    q4 = g_shift(c, 189, pos, rpos) / g_shift(c, 252, pos, rpos) - 1
    f["rs_score_w"]   = 0.4 * r63 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4    # weighted four quarters
    f["rs_score_ibd"] = 0.4 * r63 + 0.2 * r126 + 0.2 * r189 + 0.2 * r252
    f["ret126"] = r126
    f["ret252"] = r252

    # Liquidity: dollar volume from the unadjusted price on that date
    turnover = p["turnover"].where(p["turnover"].notna(), c * v)
    f["dv50"] = g_roll(turnover, 50, "mean", pos)

    # ATR
    pc = g_shift(c, 1, pos, rpos)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    for n in [10, 14, 50]:
        f[f"atr{n}"] = g_roll(tr, n, "mean", pos)
    f["atr_ratio"] = f["atr10"] / f["atr50"]

    # Range tightness and pivots
    for k in PIVOT_LOOKBACKS:
        hk = g_roll(h, k, "max", pos)
        lk = g_roll(l, k, "min", pos)
        f[f"pivot_{k}"] = hk                 # buy-stop level for tomorrow
        f[f"low_{k}"] = lk                   # structural stop
        if k in config.TIGHTNESS_WINDOWS:
            f[f"tight_{k}"] = (hk - lk) / c

    # Volume ratios
    v50 = g_roll(v, 50, "mean", pos)
    f["vol_ratio"] = v / v50
    f["vol10_50"] = g_roll(v, 10, "mean", pos) / v50
    f["vol50"] = v50

    f["bars"] = pos.astype("int32")

    for col in f.columns:
        if f[col].dtype == np.float64:
            f[col] = f[col].astype(F32)
    return f


def symbol_labels(p, delisted_set):
    """Forward returns from the next day's open to the close h bars later."""
    pos = p.groupby("symbol", observed=True).cumcount()
    rpos = p.groupby("symbol", observed=True).cumcount(ascending=False)
    c, o = p["close"], p["open"]
    next_open = g_shift(o, -1, pos, rpos)
    is_delisted = p["symbol"].astype(str).isin(delisted_set).to_numpy()
    last_close = c.groupby(p["symbol"], observed=True).transform("last")

    lab = pd.DataFrame(index=p.index)
    for hz in config.FORWARD_HORIZONS:
        fut = g_shift(c, -hz, pos, rpos)
        # Delisted stocks that die inside the window exit at their last price
        fill = is_delisted & fut.isna().to_numpy() & (rpos.to_numpy() >= 1)
        fut = fut.where(~fill, last_close)
        lab[f"fwd{hz}"] = (fut / next_open - 1).astype(F32)
    hz = max(config.FORWARD_HORIZONS)
    lab[f"fwdmax{hz}"] = (g_roll_fwd(c, hz, "max", p["symbol"], pos, rpos) / next_open - 1).astype(F32)
    return lab


# =============================================================================
# CROSS-SECTIONAL RANKS
# =============================================================================

def pct_rank_1_99(score, date, eligible):
    s = score.where(eligible)
    pct = s.groupby(date).rank(pct=True)
    return np.clip(np.ceil(pct * 99), 1, 99).astype(F32)


def build_ranks(prices_tbl, feat_path):
    feats = pq.read_table(feat_path, columns=["rs_score_w", "rs_score_ibd", "ret126",
                                              "ret252", "dv50"]).to_pandas()
    date = prices_tbl.column("date").to_pandas()
    unadj = prices_tbl.column("unadj_close").to_pandas()
    listed = prices_tbl.column("listed").to_pandas() == 1
    eligible = (unadj >= config.RS_RANK_MIN_PRICE) & listed

    ranks = pd.DataFrame(index=feats.index)
    ranks["rs_rank"]     = pct_rank_1_99(feats["rs_score_w"],   date, eligible & feats["rs_score_w"].notna())
    ranks["rs_rank_ibd"] = pct_rank_1_99(feats["rs_score_ibd"], date, eligible & feats["rs_score_ibd"].notna())
    ranks["rs_rank_252"] = pct_rank_1_99(feats["ret252"],       date, eligible & feats["ret252"].notna())
    ranks["rs_rank_126"] = pct_rank_1_99(feats["ret126"],       date, eligible & feats["ret126"].notna())
    # Dollar-volume percentile 0-100 across every listed stock trading that day
    ranks["dv_pct"] = (feats["dv50"].where(listed).groupby(date).rank(pct=True) * 100).astype(F32)
    return ranks


# =============================================================================
# MARKET FEATURES (index trend, distribution days, breadth)
# =============================================================================

def market_features(market, prices_tbl, feat_path):
    m = market.sort_values("date").reset_index(drop=True).copy()
    out = pd.DataFrame({"date": m["date"]})
    for pfx in ["spx", "comp"]:
        if f"{pfx}_close" not in m:
            continue
        c = m[f"{pfx}_close"]
        out[f"{pfx}_close"] = c
        out[f"{pfx}_sma50"] = c.rolling(50).mean()
        out[f"{pfx}_sma200"] = c.rolling(200).mean()
        out[f"{pfx}_above50"] = (c > out[f"{pfx}_sma50"]).astype("int8")
        out[f"{pfx}_above200"] = (c > out[f"{pfx}_sma200"]).astype("int8")
        out[f"{pfx}_sma50_rising"] = (out[f"{pfx}_sma50"] > out[f"{pfx}_sma50"].shift(5)).astype("int8")
        # Distribution day: close down >= 0.2% on higher volume than the day before.
        # Index volume can be missing in early years; fall back to the ETF proxy.
        vol = m.get(f"{pfx}_volume")
        proxy = {"spx": "spy_volume", "comp": "qqq_volume"}[pfx]
        if vol is None or vol.isna().all() or (vol.fillna(0) == 0).mean() > 0.5:
            vol = m.get(proxy)
        if vol is not None:
            chg = c.pct_change()
            dd = ((chg <= -0.002) & (vol > vol.shift(1))).astype("int8")
            out[f"{pfx}_dist_days25"] = dd.rolling(25, min_periods=1).sum().astype("int8")

    # Breadth: % of stocks above their own 200-day
    sma200 = pq.read_table(feat_path, columns=["sma200"]).column("sma200").to_pandas()
    close = prices_tbl.column("close").to_pandas()
    date = prices_tbl.column("date").to_pandas()
    valid = sma200.notna()
    above = (close > sma200)[valid]
    breadth = above.groupby(date[valid]).mean() * 100
    out = out.merge(breadth.rename("pct_above200").reset_index(), on="date", how="left")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    prices_tbl = pq.read_table(config.PRICES_FILE)
    symbols = pq.read_table(config.SYMBOLS_FILE).to_pandas()
    delisted_set = set(symbols.loc[symbols["delisted"], "symbol"])
    sym_col = prices_tbl.column("symbol").to_pandas().astype(str).to_numpy()
    n = len(sym_col)

    # Chunk boundaries on symbol changes so no symbol is split
    change = np.flatnonzero(sym_col[1:] != sym_col[:-1]) + 1
    starts = np.concatenate([[0], change])
    bounds = list(starts[::args.chunk]) + [n]
    print(f"{n:,} rows, {len(starts):,} symbols, {len(bounds) - 1} chunks")

    feat_writer = lab_writer = None
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        p = prices_tbl.slice(a, b - a).to_pandas()
        p["symbol"] = p["symbol"].astype(str)
        f = symbol_features(p)
        lab = symbol_labels(p, delisted_set)
        ft, lt = pa.Table.from_pandas(f, preserve_index=False), pa.Table.from_pandas(lab, preserve_index=False)
        if feat_writer is None:
            feat_writer = pq.ParquetWriter(config.FEATURES_FILE, ft.schema)
            lab_writer = pq.ParquetWriter(config.LABELS_FILE, lt.schema)
        feat_writer.write_table(ft)
        lab_writer.write_table(lt)
        print(f"  chunk {i + 1}/{len(bounds) - 1}  rows {a:,}-{b:,}  ({time.time() - t0:.0f}s)")
    feat_writer.close()
    lab_writer.close()

    print("Ranking RS and liquidity across the market...")
    ranks = build_ranks(prices_tbl, config.FEATURES_FILE)
    ranks.to_parquet(config.RANKS_FILE, index=False)

    print("Market features...")
    market = pq.read_table(config.MARKET_FILE).to_pandas()
    market_features(market, prices_tbl, config.FEATURES_FILE).to_parquet(
        config.MARKET_FEATURES_FILE, index=False)

    print(f"Done in {time.time() - t0:.0f}s")
    print("Next: hand-check five tickers against TradingView with check_features.py")


if __name__ == "__main__":
    main()
