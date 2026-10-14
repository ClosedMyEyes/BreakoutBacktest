"""
Report helpers shared by the long and short test scripts
========================================================
Mode A statistics by date, and the Mode B tables printed next to the random
baseline (F0 on the long side, S0 on the short side).
"""

import numpy as np
import pandas as pd

import config


def by_date_stats(df, mask, col):
    """Per-day mean of `col` for rows in mask, then the time series."""
    sub = df.loc[mask, ["date", col]].dropna()
    return sub.groupby("date")[col].mean()


def tstat_overlap(series, h):
    """t-stat of the mean of a daily series of h-day forward returns, which
    overlap. Newey-West standard error with h lags, so overlapping days don't
    count as independent. Returns (t, number of non-overlapping h-day periods)."""
    x = series.dropna().to_numpy(np.float64)
    n = len(x)
    n_indep = n // max(1, h)
    if n_indep < 3:
        return None, n_indep
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(h, n - 1) + 1):
        var += 2 * (1 - k / (h + 1)) * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return None, n_indep
    return float(x.mean() / np.sqrt(var / n)), n_indep


def spx_forward(market):
    m = market.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": m["date"]})
    for h in config.FORWARD_HORIZONS:
        out[f"spx_fwd{h}"] = m["spx_close"].shift(-h) / m["spx_open"].shift(-1) - 1
    return out


def block_table(trades, f0, years=5, base="f0"):
    """Average R of the family and of F0 / S0 (all seeds) for each block of years."""
    def by_block(t):
        y = pd.to_datetime(t["entry_date"]).dt.year
        return t["result_R"].groupby((y // years) * years)
    fam, rnd = by_block(trades), by_block(f0)
    out = pd.DataFrame({"trades": fam.size(), "avg_R": fam.mean(), f"{base}_avg_R": rnd.mean()})
    out["edge_R"] = out["avg_R"] - out[f"{base}_avg_R"]
    out.index = [f"{b}-{b + years - 1}" for b in out.index]
    return out.round(3)


def exit_table(trades, f0, base="f0"):
    """Count, average R and win rate for each exit reason."""
    def one(t):
        g = t.groupby("exit_reason")["result_R"]
        return pd.DataFrame({"trades": g.size(), "avg_R": g.mean().round(2),
                             "win_rate": (g.apply(lambda r: (r > 0).mean()) * 100).round(1)})
    out = one(trades)
    if f0 is not None and len(f0):
        out = out.join(one(f0).add_prefix(f"{base}_"), how="outer")
    return out.sort_values("trades", ascending=False)
