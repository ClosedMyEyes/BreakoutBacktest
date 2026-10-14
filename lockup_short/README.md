# Lock-up expiry short

Idea #3 from the project's `ideas/next_setups.md`. Insiders and early investors are usually barred from
selling for 180 days after an IPO. When the lock-up ends, a big block of new supply can hit the market on
a date that was known from the listing day. Does the stock fall around that day, by enough to short it
after costs?

`lockup_events.py` is an event table (no stops or sizing yet):

- **Listing day:** the data has no IPO dates, so each stock's first bar stands in for it. Stocks whose
  history starts with the data, returns to an exchange (`SYMBOL~2`) and SPAC-like starts (printed close
  stuck at about $10) are left out. Spin-offs and OTC uplistings can't be told apart from IPOs.
- **Day 0:** the first trading day 180 calendar days after the first bar (`--days 90` for short lock-ups).
- **Trades:** short the close of day A, cover the close of day B, for every A in -20/-10/-5/-3/-1/0 and B in
  0/1/3/5/10/20. The pick, set before the first run, is `-5:5`.
- **Universe:** judged at the day -20 close: $5, top 80% by dollar volume, no recent price break.
- **Costs:** round trip plus 5%/yr borrow (`short`), the same hedged with the S&P (`hedged`), and at
  100%/yr borrow (`b100`).

```
python lockup_short/lockup_events.py --start 2010-01-01
python lockup_short/lockup_events.py --start 2010-01-01 --days 90
python lockup_short/lockup_events.py --start 2010-01-01 --pick=-3:1 --sample 10
python lockup_short/lockup_events.py --start 2010-01-01 --pick=-20:10 --min-dv 20 --market above200 --refine
python lockup_short/lockup_events.py --start 2010-01-01 --frozen
python lockup_short/lockup_events.py --start 2010-01-01 --frozen --universe holdout --unlock-holdout
python -m pytest lockup_short
```

The run also prints a volatility check around day 0 and the pick split by size (price, dollar volume,
listing-month dollar volume), run-up since listing and volume around day 0. Add
`--min-price 1 --liq-top-pct 100` to include the small, thin stocks the default universe leaves out.

The "does the date matter?" table repeats the windows with day 0 at 90, 120, 150, 180, 210, 240, 300 and
365 days after the first bar, to tell a lock-up effect from young stocks lagging in general.

`--refine` prints only the tables for settling on one rule before the holdout, for the pick and the
`--min-dv` sample: earlier entries (day -60 to -5, covering at day 0 and at the pick's cover), the
dollar-volume cut ($0 to $100M a day), the S&P at the entry (above or below its 200-day, uptrend or
not), stops at 10/15/25/40% above the entry, how the gain builds day by day from entry to cover (a steady
bleed or a few big drops, and how far winners went against the short first), and the 10 worst trades to
check on a chart. Each row is
judged at the entry close, so an entry before day -20 (`--pick=-40:10`) moves the universe check there.

## Frozen rule

Frozen on 2026-10-14 on the Dev half, before its one run on the locked half (`--frozen`):

- **Stocks:** new listings (first bar, not a relist or a SPAC-like start) trading $20M+ a day (50-day
  average), $5+, in the top 80% by dollar volume, at the close 20 trading days before day 0.
- **Day 0:** the first trading day 180 calendar days after the first bar.
- **Market:** only while the S&P 500 is above its 200-day at that close.
- **Trade:** short at that close (day -20), cover at the close of day +10, no stop.

Dev since 2010: 242 trades (14.6 a year), +5.00% a trade after costs and 5%/yr borrow (t 2.89), +6.13%
hedged with the S&P (t 3.85), 12 of 17 years up. Every stop tried (10-40%) made it worse, and the gain
grew smoothly with dollar volume. Pass bar on the locked half, set before it runs: the short clearly above
0 after costs, month-clustered t above 2, and positive in more than half the years.

The full run ends with the 5 worst trades, losses and the account's return by year, and an account test
(`--sizes 5,10`: each short is that % of the account when it opens, marked at every close) with the worst
drop from a high.
