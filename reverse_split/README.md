# Reverse-split short

**Status: built, waiting for its first Dev run on the Norgate data.** The pick and the pass bar below were
set on 2026-10-14, before any run.

Idea #2 from the project's `ideas/next_setups.md`. Most reverse splits are forced. A Nasdaq stock that
closes under $1 for 30 business days gets a deficiency notice and about 180 days to fix it, and a reverse
split is the usual fix. Most of these companies burn cash and sell new shares into any strength once the
price is back up (offerings, at-the-market programs, warrant exercises). Do the stocks keep falling after
the split, by enough to pay a high borrow fee?

What the project already saw: the spike short made +11.4% a trade with a reverse split in the past year
(64 trades) against +8.6% without, and the crash short +5.83% (122 trades) against +0.88% for all trades.
Both splits were found after the fact on small samples, so they are a reason to test, not a result.

`reverse_split_events.py` is an event table (no stops or sizing):

- **Finding the splits:** the data has no corporate-actions table. Norgate's factor (printed close over
  adjusted close) changes only on an adjustment day, and jumps up by the ratio on a reverse split (10x for a
  1-for-10). A forward split, spin-off or special distribution moves it down. Day 0 is a listed bar where
  the factor rose 1.4x or more (`--min-ratio`), the same test as `config.BREAK_FACTOR_JUMP`. The stock needs
  60 bars before day 0. Only a stock's first reverse split in 60 bars counts (`--dedupe`), and a jump that
  is undone within 5 bars is a bad print and is left out.
- **Universe:** judged at the day 0 close with the printed price: $1+ (`--min-price`), top 80% by 50-day
  dollar volume (`--liq-top-pct`), not pinned. No price break in the 252 bars before day 0, judged at day -1,
  so a split day that also moved 25%+ (a break on day 0 itself) is allowed and flagged.
- **Trades:** short with no stop. In at the day 0 close (`close0`), the day 1 open (`open1`), the day 5
  close (`close5`) or the day 20 close (`close20`). Cover at the close 5, 20, 60, 120 or 250 trading days
  after the entry close (`open1` counts from the day 0 close). A stock that stops trading is covered at its
  last exchange close. A window across a downward price break is left out, an upward one counts as a real
  loss, and a trade still open when the data ends is left out.
- **Costs:** `ce.round_trip_cost()` (about 0.50%) plus the borrow fee for every trading day held (days /
  252) at 5, 25, 50, 100 and 200% a year (`--borrow-pcts`). Each row also prints the break-even fee: the fee
  at which its average short is 0. The pick gets a 1.5% round-trip stress row for the wider spreads on $1-5
  stocks.
- **Hedged:** against the S&P (short the stock, buy the S&P) and against the Nasdaq Composite. Neither is a
  real small-cap hedge.

## The pick and the pass bar (set 2026-10-14, before the first run)

**Pick (`--pick close0:60 --pick-fee 50`):** short at the day 0 close and cover at the close of day 60, no
stop. Universe: $1 printed, top 80% by dollar volume, first reverse split in 60 bars. Judged at **50% a year
borrow**. Sixty trading days is about three months: long enough for the offerings that usually follow a
split, and at 50% a year the borrow costs about 12%.

**Pass bar on Dev (since 2010), all four:**

1. The pick's ret is above 0 after costs and 50% a year borrow, with a month-clustered t above 2.
2. Positive in more than half the years, and in each of the 2015-19, 2020-24 and 2025+ blocks.
3. Big enough to matter: about 5% a trade, or a smaller edge with about 50+ trades a year.
4. Still positive with day 0 from March 2020 to December 2021 left out.

The script prints each check with its value. If Dev passes, the rule gets frozen here, logged, and run once
with `--universe holdout --unlock-holdout`, with no changes after that run. Every other row is context:
the run prints how many tries it holds (20 entry x hold rows at 5 fees, plus the split buckets) and logs
that count in the research log.

Two questions in the test plan were left open and are set to these defaults until Mark says otherwise:

- **Price floor: $1 printed.** The usual $5 floor would drop most of these stocks (a $0.40 stock after a
  1-for-10 prints $4). The pick's table shows the result by price, and `--min-price 2` or `5` reruns it.
- **Borrow fee: 50% a year.** It could be checked against live IBKR fees on a few recent reverse-split
  stocks. The fee ladder and the break-even fee show what any other fee would give.

## What the run prints

1. Counts: factor jumps found, bad prints, repeats, too little history, inside the dates, the universe
   (and how many failed each rule), events a year, the ratio distribution and the most common ratios, and
   the printed price on day -1 and day 0. Then two data checks: whether the day 0 and day 1 opens line up
   with the adjusted closes, and whether Norgate's volume is split-adjusted.
2. Day by day from day -60 to +120 against the S&P: the average move in the short's favour and its running
   total from the day 0 close, the share moving the short's way, median volume against usual, the share
   back under $1 printed and the share delisted by that day.
3. Every entry and hold at 5% a year, then the fee ladder with the break-even fee.
4. The pick split by what was known at the day 0 close: ratio, printed price on day -1 and day 0, dollar
   volume, S&P or Russell 2000 member, the 60-day move before day 0, the day 0 move itself, another reverse
   split in the past 3 years, day 0 marked as a price break, the stock and the S&P against their 200-days.
5. The cost stress and delisting lines: how many pick trades stopped trading, their average, split by a last
   printed price under $1 (likely to OTC or bankrupt) or $1+ (often a buyout).
6. The pick by 5-year block and by year, and without March 2020 to December 2021.
7. Losses and drawdown: the 10 worst trades, losses by year, and an account test at 2% and 5% of the
   account a short (`--sizes`), marked at every close, with the worst drop and the most shorts open at once.
   It reuses `account()` and `account_summary()` from `lockup_short/lockup_events.py`.
8. `--sample N` random pick trades to check on a chart, then the pass bar with each check's value.

```
python reverse_split/reverse_split_events.py --start 2010-01-01
python reverse_split/reverse_split_events.py --start 2010-01-01 --sample 10
python reverse_split/reverse_split_events.py --start 2010-01-01 --min-price 5
python reverse_split/reverse_split_events.py --start 2010-01-01 --only ratio:5:20   # one bucket, by block
python -m pytest reverse_split
```

Later, and only once the rule is frozen:

```
python reverse_split/reverse_split_events.py --start 2010-01-01 --universe holdout --unlock-holdout
```

## What the data can't show

- **The announcement and the vote.** Only the effective date (the first bar at the new price) is in the
  data. The split is announced days to months earlier, so a short at the day 0 close needs no look-ahead,
  but any move before the announcement can't be told apart from the rest of the run-up.
- **Why the company split.** The $1 listing rule, an uplisting or something else all look the same.
- **Borrow.** Norgate has no borrow fees, shares available or short interest. The test charges an assumed
  fee from the day of entry. Many of these stocks are hard to borrow or can't be borrowed at all on a given
  day, and fees of 100%+ a year are common.
- **Share counts and offerings.** There is no direct measure of dilution.
- **Moves to OTC.** A stock that drops to OTC is covered at its last exchange close. The real OTC price is
  usually lower, so this understates those shorts. The run counts them.

Real-world risks the backtest can't model:

- **Re-delisting.** Many of these stocks fall back under $1 within a year. That is the edge, but a broker
  may force a cover, recall the shares or stop lending as the stock heads back to the deficiency list or
  to OTC.
- **Squeezes.** Low-float stocks after a reverse split are prone to sharp squeezes. Shorts here can lose
  more than 100% of the position, and a recalled borrow during a squeeze forces a cover at the worst time.
  Size for the tail, not the average.
