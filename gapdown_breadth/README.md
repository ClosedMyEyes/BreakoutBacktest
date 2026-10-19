# Market-wide gap-downs

**Status: written 2026-10-19, not yet run on the Norgate data.** The holdout stays locked until a rule is
frozen from the Dev run.

A lead from the Stockbot review (section 9 of `stockbot_review/verification/README.md` in the project
files). Stockbot's gap_fill_large buys a large cap that opened 2.5-5% below its prior close once it turns
up. On its 48 tickers' 1-minute bars since 2012, it made +0.36R a trade on mornings when 5%+ of the 48
gapped down 2.5%+ together, and lost 0.21R a trade on lone gap-downs. Buying every 2.5%+ gap-down at
09:31 and selling at the close showed the same split: +0.26% a trade on the broad days and -0.30% on the
lone ones, after a 0.10% round trip. Three things keep it a lead and not a result: the whole basket of 48
rose almost as much on the broad days (+0.20%), so it may be the market bouncing back and not the stocks
that gapped; before 2012 the split made little difference; and the 48 are companies that are still large
today. This tests it on every S&P 500 stock, delisted ones included.

`breadth_events.py` is an event table (no stops or sizing):

- **Breadth:** the share of S&P 500 members (at the prior close) that opened 2.5%+ below their prior
  close, known once they have opened. A price-break day (a bad split adjustment) doesn't count, and days
  with fewer than 100 members are left out (`--min-members`). It is read from the loaded half only, so on
  Dev it covers about half the index and never touches the locked half. A day with 5%+ is broad
  (`--broad`); the rest are lone days.
- **Event:** a stock that passed the universe at the prior close ($5, top 50% by dollar volume, no price
  break in the past year, and an S&P 500 member: `--size sp500`, the default here) opens 2.5-5% below
  that close (`--min-gap`, `--max-gap`), with no price break on the day.
- **Trade:** buy at the open, sell at the close. `ret` is after a 0.10% round trip (`--cost-pct`), a
  little above the 0.08% the review's `costs.py` charges a $50 large cap with IBKR's commission.
  `--cost-pct 0.5`, the round trip the repo's other event studies use (0.20% slippage a side and
  commission at a $10 stock), is a stress. `gross` is before costs, and `vs_spy` is the gross minus SPY's
  open-to-close that day. SPY itself is shown by the same breadth, after 0.02% (`--spy-cost-pct`).
- **The pick, set before the first run:** the events on broad days, 2010 on. It's worth building on if
  `ret` is clearly above 0 with a month-clustered t above 2, positive in more than half the years and in
  the recent 5-year blocks, and above the lone days' `ret`. The other bins, the basket and the splits are
  context, not more picks.

Breadth is only known once the stocks have opened, so a live trade would buy a moment after the open,
but the daily bars only have the open. On the 48 tickers' minute bars that doesn't flatter the trade: for
2.5-5% gaps on broad days since 2012, the first minute after the open fell 0.09% on average (open to
close +0.35% before costs, 09:31 to close +0.44%). Before 2012 it was about 0.

```
python gapdown_breadth/breadth_events.py
python gapdown_breadth/breadth_events.py --sample 10 --save-trades gapdown_trades.csv
python gapdown_breadth/breadth_events.py --cost-pct 0.5
python gapdown_breadth/breadth_events.py --size all --max-gap 100
python -m pytest gapdown_breadth
```

The run prints, each before 2010 and 2010 on:

1. How often each breadth level comes up, and the number of events.
2. The events by breadth: under 5%, 5-10%, 10-20%, 20-30%, 30%+, and broad (5%+).
3. SPY bought at the open and sold at the close, by the same breadth.
4. Every S&P 500 member's open-to-close that day, gap or not, by breadth: the basket the events are
   measured against. If the events only match it, the gain is the market's and not the gap's.
5. The pick next to the lone days, split by gap size and S&P index (both on any stock that passed the
   universe), by the S&P against its 200-day, and by price.
6. The pick and the lone days by 5-year block and by year (2010 on), with SPY on the broad days.
7. The pick's line and the pass bar.

`--sample 10` prints ten random pick trades to check on a chart, and `--save-trades` writes every event
trade to a CSV. Each run adds a row to the research log unless `--no-log` is given.

If the pick passes on Dev, the next step is to freeze it and run it once on the locked half
(`--universe holdout --unlock-holdout`). A pass there would make it a strategy of its own on daily data,
and a reason to test a breadth filter for gap_fill_large on the March to October 2026 bars before the bot
uses one.
