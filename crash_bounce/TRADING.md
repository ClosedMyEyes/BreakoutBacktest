# Trading the bounce short

The frozen rule (`prev:touch:1:0`, hold 10, no stop; holdout passed 2026-10-13):

- **Crash day:** a stock closes 30%+ below its close 3 days earlier, the first such day in 10, after a 60-day rise of 20%+ before the fall. It needs a $5+ close and a place in the top 80% of stocks by dollar volume.
- **Entry, the next session only:** a limit order to short at the close the day before the crash day. If the stock opens above it, it fills at the open. If the high only reaches it, it fills at the limit. If it never gets there, there's no trade.
- **Exit:** no stop. Cover at the market close on the 10th trading day after the fill, counting the day you short as day 0.

## Each evening

Run the bounce scan after the spike scan, on the same refreshed data:

```
python pipeline/build_data.py
python pipeline/build_features.py
python spike_short/spike_scan.py
python crash_bounce/bounce_scan.py
```

It prints two tables.

1. **Orders for the next session**: each stock that crashed today, with `short_limit`, the price to short at. `limit_over_close_%` is how far it has to bounce from today's close to fill. Most evenings this is empty, since the backtest filled about 22 a year.
2. **Shorts entered in the last 15 trading days**: each short, filled at the open or at the limit, with its price, `days_left` to the cover and where it stands.

Both go into `bounce_signals.csv` in the work folder, the forward-test record. `--asof 2026-08-07` shows what it would have said that evening, without saving.

Before the open, check that each stock on the order list can be borrowed.

## The next session

- **The order:** a day limit order to sell short at `short_limit`, regular hours only (not pre-market). If the stock opens above the limit, it fills at the open, as in the backtest. If it doesn't trade up to the limit by the close, the order expires and the setup is gone.
- **Short-sale restriction:** the crash usually puts the stock under SSR that day. SSR still lets a short limit order fill above the bid, which is what this order is. Write down any order the broker rejects or fills differently, since that changes the results.
- **No stop.**
- **The cover:** the evening `days_left` shows 1, cover at the next session's close with a market-on-close order.

## On TradingView

`bounce_signal.pine` is the same rule as a daily-chart indicator. Add it once: open the Pine Editor, choose New, then Indicator, paste in all of `bounce_signal.pine`, save it as "BounceShort" and click Add to chart.

It marks:

- **Crash days:** a purple triangle under the bar and a purple shade over it, with a dotted line at the limit for the next session. The shade can be turned off under Settings, then Style ("Crash day background").
- **Entries the backtest took:** a red S, with a dashed line at the fill price out to the cover.
- **Covers:** a gray x on the 10th bar after the entry.
- **A crash on the last bar:** a label with the limit for the next session.

Set "Lowest 50-day average dollar volume" to the liquidity floor `spike_short/spike_scan.py` prints each night. Both setups use the same floor. The indicator doesn't check Norgate's bad-data bars or the backtest's rule against a price break in the past year.

Check it on trades the backtest took. Each chart should show a triangle on the crash day, an S the next session and an x on the cover:

- SEDG: crash 2026-08-07, short 2026-08-10 at 32.67, cover 2026-08-24
- VPG: crash 2026-08-07, short 2026-08-10 at 69.69, cover 2026-08-24

A stock with fewer than 63 bars on TradingView can't show a crash, because the 60-day rise needs them. HOST is one: the backtest shorted it on 2026-09-22, but TradingView's chart starts only a few days before its crash. The Norgate data has older history for it, so the nightly scan still catches cases like this.

If SEDG or VPG is missing its marks, send a screenshot. The indicator's logic was checked against the backtest in Python, and the script loads on TradingView.

## Without the Norgate data

TradingView's stock screener can't filter on a 3-day change, so a loose screen finds the candidates and the indicator applies the rule. Save a screen called "Bounce 1W" with these filters:

- Price at least 5
- Performance 1W at most -15%
- Average Volume 30D at least 20K

Run it after the close and open each chart. A purple triangle on today's bar means an order for the next session. With a Premium plan or higher, put the names in a watchlist and run the Pine Screener with this indicator on 1D, filtering on "Order (1 = short limit next session)" equal to 1.

The screen can miss a crash on a stock that jumped 20%+ in the two days before its fall started. `bounce_scan.py` on the full data misses nothing.
