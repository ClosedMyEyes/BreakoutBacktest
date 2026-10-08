"""
Data sources for build_data.py
==============================
Two interchangeable sources produce the same three tables:

  prices   one row per symbol-day: symbol, date, open, high, low, close, volume
           (split-adjusted), unadj_close, turnover, index flags (in_sp500, ...)
  symbols  one row per symbol: symbol, name, sector, industry, first_date,
           last_date, delisted
  market   one row per day: date, <prefix>_open/high/low/close/volume for each
           MARKET_SYMBOLS entry

NorgateSource   the real thing. Needs Windows, the norgatedata package and the
                Norgate Data Updater running.
SyntheticSource random but realistic-looking stocks (trends, bases, breakouts,
                splits, delistings) so every script can be built and debugged
                before the trial starts. Never use its results for decisions.
"""

import re
import zlib

import numpy as np
import pandas as pd

import config

EXCLUDE_NAME_RE = re.compile(r"\b(" + "|".join(config.EXCLUDE_NAME_WORDS) + r")\b")


# =============================================================================
# NORGATE
# =============================================================================

class NorgateSource:

    def __init__(self, start_date=config.START_DATE):
        import norgatedata   # imported here so the rest of the code runs without it
        self.nd = norgatedata
        self.start_date = start_date

    # ── Universe ─────────────────────────────────────────────────────────────
    def all_symbols(self):
        syms = []
        for db in config.NORGATE_DATABASES:
            syms += self.nd.database_symbols(db) or []
        return sorted(set(syms))

    def subtype(self, symbol, level=1):
        try:
            return getattr(self.nd, f"subtype{level}")(symbol)
        except Exception:
            return None

    def is_common_stock(self, symbol):
        """Subtype filter first (e.g. Equity), then drop names that look like
        units, warrants, preferreds and funds that slip through it."""
        if config.COMMON_STOCK_SUBTYPES is not None and \
                self.subtype(symbol) not in config.COMMON_STOCK_SUBTYPES:
            return False
        if config.EXCLUDE_SUBTYPE2 and self.subtype(symbol, 2) in config.EXCLUDE_SUBTYPE2:
            return False
        name = (self.nd.security_name(symbol) or "").upper()
        return not (EXCLUDE_NAME_RE.search(name) or "%" in name)

    # ── One symbol ───────────────────────────────────────────────────────────
    def symbol_prices(self, symbol):
        nd = self.nd
        df = nd.price_timeseries(
            symbol,
            stock_price_adjustment_setting=nd.StockPriceAdjustmentType.CAPITAL,
            padding_setting=nd.PaddingType.NONE,
            start_date=self.start_date,
            timeseriesformat="pandas-dataframe",
        )
        if df is None or len(df) == 0:
            return None
        cols = {c.lower(): c for c in df.columns}
        out = pd.DataFrame({
            "date":        pd.to_datetime(df.index),
            "open":        df[cols["open"]].to_numpy(),
            "high":        df[cols["high"]].to_numpy(),
            "low":         df[cols["low"]].to_numpy(),
            "close":       df[cols["close"]].to_numpy(),
            "volume":      df[cols["volume"]].to_numpy(),
            "unadj_close": df[cols["unadjusted close"]].to_numpy()
                           if "unadjusted close" in cols else np.nan,
            "turnover":    df[cols["turnover"]].to_numpy()
                           if "turnover" in cols else np.nan,
        })
        for col, index_name in config.INDEX_MEMBERSHIP.items():
            try:
                idx = nd.index_constituent_timeseries(
                    symbol, index_name,
                    padding_setting=nd.PaddingType.NONE,
                    start_date=self.start_date,
                    timeseriesformat="pandas-dataframe",
                )
                flag = idx.iloc[:, -1].reindex(df.index).fillna(0)
                out[col] = flag.to_numpy().astype("int8")
            except Exception:
                out[col] = np.int8(0)
        out.insert(0, "symbol", symbol)
        return out

    def symbol_meta(self, symbol, prices):
        nd = self.nd

        def safe(fn, *a):
            try:
                return fn(*a)
            except Exception:
                return None

        sector   = safe(nd.classification_at_level, symbol, config.CLASSIFICATION_SCHEME, "Name", 1)
        industry = safe(nd.classification_at_level, symbol, config.CLASSIFICATION_SCHEME, "Name", 3)
        delisted = "-" in symbol and symbol.rsplit("-", 1)[-1].isdigit()   # Norgate delisted suffix
        return {
            "symbol":     symbol,
            "name":       safe(nd.security_name, symbol),
            "subtype":    safe(nd.subtype1, symbol),
            "sector":     sector or "Unknown",
            "industry":   industry or "Unknown",
            "first_date": prices["date"].iloc[0],
            "last_date":  prices["date"].iloc[-1],
            "last_price": float(prices["unadj_close"].iloc[-1]),
            "delisted":   bool(delisted),
        }

    # ── Market series ────────────────────────────────────────────────────────
    def market(self):
        nd = self.nd
        frames = []
        for prefix, sym in config.MARKET_SYMBOLS.items():
            try:
                df = nd.price_timeseries(
                    sym,
                    stock_price_adjustment_setting=nd.StockPriceAdjustmentType.CAPITAL,
                    padding_setting=nd.PaddingType.NONE,
                    start_date=self.start_date,
                    timeseriesformat="pandas-dataframe",
                )
            except Exception as e:
                print(f"  market series {sym} unavailable: {e}")
                continue
            cols = {c.lower(): c for c in df.columns}
            part = pd.DataFrame({
                f"{prefix}_{k}": df[cols[k]] for k in ["open", "high", "low", "close", "volume"]
                if k in cols
            })
            part.index = pd.to_datetime(part.index)
            frames.append(part)
        mkt = pd.concat(frames, axis=1).sort_index()
        mkt.index.name = "date"
        return mkt.reset_index()


# =============================================================================
# SYNTHETIC
# =============================================================================

class SyntheticSource:
    """
    Random stocks for debugging. Each stock alternates between regimes
    (uptrend, base, downtrend, chop) so the Trend Template and breakout code
    have something to find. Some stocks split, some get delisted (a few to
    near zero, like bankruptcies), and a market index drives a shared factor.
    """

    SECTORS = ["Tech", "Health", "Financials", "Energy", "Industrials",
               "Consumer", "Materials", "Utilities"]

    def __init__(self, n_symbols=400, start_date="2005-01-01", end_date="2015-12-31", seed=7):
        self.n = n_symbols
        self.dates = pd.bdate_range(start_date, end_date)
        self.rng = np.random.default_rng(seed)
        self._market = self._make_market()

    def _make_market(self):
        n = len(self.dates)
        regime_drift = np.repeat(self.rng.choice([0.0008, 0.0004, -0.0010], size=n // 120 + 1,
                                                 p=[0.5, 0.3, 0.2]), 120)[:n]
        rets = regime_drift + self.rng.normal(0, 0.011, n)
        close = 1000 * np.exp(np.cumsum(rets))
        self._mkt_rets = rets
        openp = close * np.exp(self.rng.normal(0, 0.003, n))
        high = np.maximum(openp, close) * (1 + np.abs(self.rng.normal(0, 0.004, n)))
        low = np.minimum(openp, close) * (1 - np.abs(self.rng.normal(0, 0.004, n)))
        vol = self.rng.lognormal(20, 0.25, n)
        mkt = pd.DataFrame({"date": self.dates})
        for p in config.MARKET_SYMBOLS:
            mkt[f"{p}_open"], mkt[f"{p}_high"], mkt[f"{p}_low"] = openp, high, low
            mkt[f"{p}_close"], mkt[f"{p}_volume"] = close, vol
        return mkt

    def all_symbols(self):
        return [f"S{i:04d}" for i in range(self.n)]

    def is_common_stock(self, symbol):
        return True

    def subtype(self, symbol, level=1):
        return "Synthetic"

    def symbol_prices(self, symbol):
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        n_all = len(self.dates)
        start = int(rng.integers(0, n_all // 3))
        end = n_all
        delist = rng.random() < 0.25
        if delist:
            end = int(rng.integers(start + 300, n_all)) if start + 300 < n_all else n_all
        n = end - start
        if n < 60:
            return None

        # Regime-switching drift: uptrend / base / downtrend / chop
        drifts, vols = [], []
        while len(drifts) < n:
            kind = rng.choice(4, p=[0.35, 0.25, 0.2, 0.2])
            length = int(rng.integers(20, 120))
            d, v = [(0.003, 0.022), (0.0, 0.010), (-0.0025, 0.028), (0.0, 0.020)][kind]
            drifts += [d] * length
            vols += [v] * length
        drift = np.array(drifts[:n])
        vol = np.array(vols[:n]) * rng.uniform(0.7, 1.6)
        beta = rng.uniform(0.5, 1.6)
        rets = drift + beta * self._mkt_rets[start:end] + rng.normal(0, 1, n) * vol
        if delist and rng.random() < 0.4:              # bankruptcy-style collapse
            rets[-40:] -= 0.06
        price0 = rng.uniform(4, 80)
        close = price0 * np.exp(np.cumsum(rets))
        gap = rng.normal(0, 0.5, n) * vol
        openp = close * np.exp(-rets + gap)            # open near prior close plus a gap
        openp[0] = close[0]
        rng_hl = np.abs(rng.normal(0, 1, n)) * vol * 0.8
        high = np.maximum(openp, close) * (1 + rng_hl)
        low = np.minimum(openp, close) * (1 - rng_hl)
        volume = rng.lognormal(rng.uniform(11, 15), 0.4, n) * (1 + 3 * np.abs(rets) / vol.mean())

        # Unadjusted prices: one 2:1 split for some stocks. Adjusted series stays
        # smooth; unadjusted is 2x higher before the split date.
        unadj = close.copy()
        if rng.random() < 0.3 and n > 400:
            k = int(rng.integers(200, n - 100))
            unadj[:k] *= 2.0

        dates = self.dates[start:end]
        out = pd.DataFrame({
            "symbol": symbol, "date": dates,
            "open": openp, "high": high, "low": low, "close": close,
            "volume": volume, "unadj_close": unadj, "turnover": close * volume,
        })
        big = rng.random() < 0.3
        for col in config.INDEX_MEMBERSHIP:
            out[col] = np.int8(1 if (col == "in_sp500" and big) else 0)
        return out

    def symbol_meta(self, symbol, prices):
        rng = np.random.default_rng(zlib.crc32((symbol + "meta").encode()))
        return {
            "symbol": symbol, "name": f"Synthetic {symbol}", "subtype": "Synthetic",
            "sector": self.SECTORS[int(rng.integers(len(self.SECTORS)))],
            "industry": "Synthetic",
            "first_date": prices["date"].iloc[0], "last_date": prices["date"].iloc[-1],
            "last_price": float(prices["unadj_close"].iloc[-1]),
            "delisted": bool(prices["date"].iloc[-1] < self.dates[-1]),
        }

    def market(self):
        return self._market.copy()
