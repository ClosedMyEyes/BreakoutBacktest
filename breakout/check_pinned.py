"""
Buyout-pinned stocks: how to spot them before buying (Phase 5 check)
====================================================================
Once a takeover is announced the stock sits near the deal price until the
deal closes, often making tiny new highs that look like breakouts or "tight"
bases. In a trade log they show up as delisted exits near 0R.

This compares the signal-day range of those trades with every other trade,
so a minimum-range filter can be set where it drops most pinned stocks and
few real setups. Ranges are as of the signal day (no look-ahead).

HOW TO RUN:
  python breakout/setup_trend.py --save-trades results/f1_trades.csv
  python breakout/check_pinned.py results/f1_trades.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import datastore  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("trades")
parser.add_argument("--flat-r", type=float, default=0.3, help="Delisted exits within this many R count as pinned")
args = parser.parse_args()


def main():
    t = pd.read_csv(args.trades, parse_dates=["signal_date"])
    f = datastore.load_panel(["close", "tight_10", "atr10"], universe="dev")
    f["symbol"] = f["symbol"].astype(str)
    t = t.merge(f.rename(columns={"date": "signal_date"}), on=["symbol", "signal_date"], how="left")
    t["range10"] = t["tight_10"] * 100
    t["atr10_pct"] = t["atr10"] / t["close"] * 100

    delisted = t["exit_reason"] == "delisted"
    pinned = delisted & (t["result_R"].abs() < args.flat_r)
    groups = {"pinned (delisted, ~0R)": pinned, "other delisted": delisted & ~pinned, "all other trades": ~delisted}

    for col, cuts in [("range10", [1, 1.5, 2, 2.5, 3, 4, 5]), ("atr10_pct", [0.5, 0.75, 1, 1.25, 1.5, 2])]:
        print(f"\n{col} on the signal day (%): quantiles")
        q = pd.DataFrame({k: t.loc[m, col].quantile([.05, .1, .25, .5, .75]) for k, m in groups.items()})
        print(q.round(2).to_string())
        rows = []
        for x in cuts:
            below = t[col] < x
            row = {f"{col} <": x}
            for k, m in groups.items():
                row[f"% of {k.split(' (')[0]}"] = round(float((below & m).sum() / max(m.sum(), 1) * 100), 1)
            rest = below & ~delisted
            row["avg R of other trades cut"] = round(float(t.loc[rest, "result_R"].mean()), 3) if rest.any() else np.nan
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n{int(pinned.sum())} pinned of {len(t)} trades; overall avg R {t['result_R'].mean():.3f}")


if __name__ == "__main__":
    main()
