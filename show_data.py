"""
Look up the raw data (Phase 1 hand-checks)
==========================================
Prints what build_data.py saved for one stock, or searches symbols.parquet,
so you don't need to open Parquet files by hand.

HOW TO RUN:
  python show_data.py ABPO              # symbol info + first/last bars + every adjustment day
  python show_data.py ABPO --rows 40    # more bars from the start of its history
  python show_data.py --find lehman     # search names and tickers
  python show_data.py --delisted        # every delisted stock with its last date and price

Columns in the bar table:
  close        adjusted close (what all features use)
  unadj_close  the price that actually printed that day
  factor       unadj_close / close; it changes only on a split or other capital event
  adj_chg%     day-over-day change in the adjusted close (should look normal on a split day)
  unadj_chg%   day-over-day change in the printed price (jumps on a split day)
  listed       1 = on NYSE/Nasdaq/NYSE American that day, 0 = OTC or before listing
  note         "<- PRICE BREAK" = old and new prices don't connect (see config.BREAK_*);
               the stock gets no new entries for a year after it
"""

import argparse

import pandas as pd
import pyarrow.parquet as pq

import config
import rules

parser = argparse.ArgumentParser()
parser.add_argument("symbol", nargs="?")
parser.add_argument("--rows", type=int, default=15, help="Bars to show from the start of the history")
parser.add_argument("--find", type=str, default=None, help="Search symbol names and tickers")
parser.add_argument("--delisted", action="store_true", help="List every delisted stock")
args = parser.parse_args()

pd.set_option("display.width", 200)


def load_symbols():
    s = pd.read_parquet(config.SYMBOLS_FILE)
    s["symbol"] = s["symbol"].astype(str)
    return s


def bars(sym):
    t = pq.read_table(config.PRICES_FILE, filters=[("symbol", "==", sym)],
                      columns=["date", "open", "high", "low", "close", "unadj_close", "volume", "listed"])
    p = t.to_pandas().sort_values("date").reset_index(drop=True)
    p["factor"] = p["unadj_close"] / p["close"]
    p["adj_chg%"] = p["close"].pct_change() * 100
    p["unadj_chg%"] = p["unadj_close"].pct_change() * 100
    jump = p["factor"] / p["factor"].shift(1)
    p["note"] = ""
    p.loc[(jump > 1.01) | (jump < 0.99), "note"] = "<- adjustment"
    brk = rules.price_breaks(pd.Series(sym, index=p.index), p["close"], p["unadj_close"])
    p.loc[brk, "note"] = "<- PRICE BREAK"
    return p


def fmt(df):
    return df.to_string(index=False, float_format=lambda x: f"{x:.4g}")


def show_symbol(sym, symbols):
    info = symbols[symbols["symbol"] == sym]
    if info.empty:
        near = symbols[symbols["symbol"].str.startswith(sym.split("-")[0])]["symbol"].tolist()
        print(f"{sym} is not in symbols.parquet." + (f" Close matches: {', '.join(near[:10])}" if near else ""))
        return
    print(f"=== {sym} ===")
    print(info.T.iloc[:, 0].to_string())

    p = bars(sym)
    print(f"\nFirst {min(args.rows, len(p))} of {len(p)} bars:")
    print(fmt(p.head(args.rows)))

    adj = p.index[p["note"] != ""]
    if len(adj):
        print(f"\nAdjustment days and price breaks ({len(adj)}), with the bar before and after:")
        for i in adj:
            print(fmt(p.iloc[max(0, i - 1): i + 2]))
            print()
    else:
        print("\nNo adjustment days.")

    print("Last 5 bars:")
    print(fmt(p.tail(5)))
    print(f"\nTradingView: https://www.tradingview.com/chart/?symbol={sym.split('-')[0]}")


def main():
    symbols = load_symbols()
    if args.find:
        q = args.find.lower()
        hit = symbols[symbols["symbol"].str.lower().str.contains(q, regex=False)
                      | symbols["name"].fillna("").str.lower().str.contains(q, regex=False)]
        print(fmt(hit[["symbol", "name", "sector", "first_date", "last_date", "last_price", "delisted"]])
              if len(hit) else f"Nothing matches {args.find!r}")
    elif args.delisted:
        d = symbols[symbols["delisted"]].sort_values("last_date")
        print(fmt(d[["symbol", "name", "first_date", "last_date", "last_price"]]))
        print(f"\n{len(d)} delisted of {len(symbols)} symbols")
    elif args.symbol:
        show_symbol(args.symbol, symbols)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
