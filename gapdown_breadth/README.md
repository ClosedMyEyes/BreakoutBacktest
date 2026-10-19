# Market-wide gap-downs

**Status: the pick failed on Dev (Mark's run, 2026-10-19; below).** `filters.py` looks at where the trades
lose and at filters on the stock itself; it hasn't been run yet. The holdout stays locked.

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

## Dev result (Mark's run, 2026-10-19): the pick failed

9,260 days had a breadth reading (1990-01-03 to 2026-10-09, a median 262 members a day); 93.9% of them were
under 5%. There were 33,582 events on 6,400 days.

- **The pick, 2010 on:** 7,607 trades on 184 days, **-0.11% a trade** after the 0.10% round trip (t -0.48),
  -0.04% against SPY, positive in 6 of 15 years. The lone days made +0.04%, and SPY itself +0.10% a day on
  the broad days (t 1.09).
- **By breadth, 2010 on:** 5-10% -0.04% (1,338 trades), 10-20% +0.71% (1,425, t 2.16), 20-30% -0.60% (674),
  30%+ -0.33% (4,170). So the 50 crash days (20%+) held 4,844 of the 7,607 trades and lost 0.37% each,
  while the 5-20% days made +0.35% on 2,763. 2011 and 2025 lost the most; 2020 was about flat (-0.07%).
- **Before 2010 it worked on any day:** broad days +0.45% (t 2.01), lone days +0.39% (t 7.41), and the crash
  days were the best (+0.50% on 4,508 trades). By 5-year block the broad days made +0.58%, +1.20% and +0.74%
  from 1990 to 2004, then -0.06% (2005-09), -0.42%, +0.24%, -0.00% and -0.32% (2025-26); the lone days
  +0.59%, +0.34%, +0.18% and +0.48% to 2009, then +0.10%, -0.04%, -0.05% and +0.27%. Buying gap-downs at
  the open was an edge before 2010 on any day, and has mostly faded since: not a market-breadth effect.

"Skip the crash days" would turn 2010 on positive, but it was picked after seeing the result, it was the
best rule before 2010, and the locked half can't check it: it holds other companies on the same days. So it
isn't taken.

## Part 2: where it loses, and filters on the stock

`filters.py` uses the same events, breadth and costs (and flags). It prints:

1. Broad and lone days by year before 2010 (`breadth_events.py` printed 2010 on), for 2008 and 2000-02.
2. The 15 worst and 15 best days over all years, by the day's events' ret added up, with the day's breadth
   and SPY's gap and open-to-close; then 2010 on without the 10 worst and without the 10 best days.
3. Every event from 2010 split by four things about the stock, all known at the open, with broad and lone
   days side by side, ret by 5-year block, and ret before 2010:
   - **Trend:** the prior close above or below the stock's 200-day, and its 50-day, average. A gap-down in
     a rising stock may get bought back; in a falling one it may keep going.
   - **Gap below SPY's:** how many points further the stock gapped down than SPY that morning (under 1, 1 to
     2.5, 2.5+). A gap close to the market's is mostly the market's move; one far below it is mostly the
     stock's own news, which tends to keep going.
   - **Gap over its 50-day ATR** (as % of the prior close): under 1x, 1x to 2x, 2x+. 3% is a shock for a
     steady stock and an ordinary day for a jumpy one.
   - **The day before:** fell 2%+, -2% to +2%, rose 2%+. A second fall in a row may be panic that turns, or
     a slide that carries on.

**The bar, set before the first run:** a bucket is worth one run on the locked half if, on all days from 2010,
its ret is above 0 with a month-clustered t above 2, and above 0 in each 5-year block from 2010. There are 13
buckets, so one can clear t 2 by luck, and the locked half shares the days, so even a pass there would need
the forward test on the March to October 2026 minute bars before the bot uses it. A rule about which days to
trade (breadth, SPY's gap) stays context. Each run adds a row to the research log with 13 combinations.

```
python gapdown_breadth/filters.py
python gapdown_breadth/filters.py --save-trades gapdown_filters.csv
```

`--save-trades` writes every event with its filter values (`vs200`, `vs50`, `below_spy`, `gap_atr`,
`day_before`, `spy_gap`), so other splits can be read off the file without another run.
