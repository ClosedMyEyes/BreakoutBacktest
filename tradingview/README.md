# The gap-up short on TradingView

`spike_signal.pine` is the frozen rule from `spike_short.py` as a daily-chart indicator. It marks:
- **Signal days**, with an orange triangle. These are the closes that put a stock on the next morning's watch list.
- **Entries the backtest would have taken**, with a red S and a dashed stop line 25% above the open, drawn for 10 bars.

On the last bar, a label gives the price the next open has to beat.

It leaves out two of the backtest's checks: the bottom-20% dollar-volume filter, and Norgate's bad-data bars. Almost every spike clears the dollar-volume filter anyway, because its volume explodes.

## Add it (once)

1. Open any daily chart, then the Pine Editor at the bottom. Choose New, then Indicator. Paste in all of `spike_signal.pine`, save it as "Spike short signal" and click Add to chart.
2. Check it on trades the backtest took. Each of these charts should show an S on the entry date:
   - GRML, 2026-09-22
   - AXTI, 2026-08-07
   - FTFT, 2026-09-16

   If one is missing or the editor shows an error, send a screenshot. The logic was checked against the backtest in Python, but there was no Pine compiler to test the script itself.

## Find the candidates each evening

TradingView's stock screener can't filter on "100% above the lowest close of the last 10 days", so this takes two steps. First, a loose screen finds the candidates. Then the indicator applies the exact rule.

1. In Screeners, then Stocks, set the market to USA. Save two screens:
   - "Spike 1W": Price at least 5 and Performance 1W at least 30%. This catches nearly every run that's still going.
   - "Spike 1M": Price at least 5 and Performance 1M at least 80%. This catches stocks that doubled early in the 10 days and drifted after.
2. Run both after the close. For each name, open its chart. An orange triangle on today's bar means it's on tomorrow's list, and the label gives the price to beat.
   - With a Premium plan or higher, the Pine Screener can do this check in one pass. Put the candidates in a watchlist and run the Pine Screener on it with this indicator on 1D. Then filter on "Watch (1 = short next open if it gaps)" equal to 1, or on the "Spike watch" alert condition.
3. Post the list in the project thread: ticker, today's close and the price to beat. Claude builds the news dossiers and runs the three blind judges before the open.

The two screens can still miss a stock that fell into its low and doubled more than a week ago. `spike_scan.py` on the full data misses nothing, so use it while the Norgate data is still current.

## Next morning

- **Entry:** short only if the stock opens above the price to beat. Around 9:25 ET, look at the premarket price.
  - If it's well above the line, send the short as an on-open order.
  - If it's close to the line, wait for the official open print, then short.
- **Stop:** 25% above the fill.
- **Exit:** cover at the close of the 10th trading day.
