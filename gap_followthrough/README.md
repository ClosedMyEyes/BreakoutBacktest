# Big-gap follow-through

**Status: parked (2026-10-14).** No edge on Dev in either direction, and none of the lead-up filters
turned it around (see "Second Dev run" below). The holdout was never used.

Idea #4 from the project's `ideas/next_setups.md`. A liquid stock that opens 5%+ away from the prior close
on 3x+ its usual volume has nearly always had news, usually earnings. Prices drifting on in the direction
of an earnings surprise is one of the oldest documented edges, but it is reported to have shrunk in large
caps since 2010. Does going with the gap still pay after costs?

`gap_events.py` is an event table (no stops or sizing yet):

- **Gap:** day 0 opens 5%+ above or below the prior close (`--min-gap`) on 3x+ the stock's average volume
  over the 50 days before (`--min-vol-x`). Only a stock's first such gap in 20 bars counts (`--dedupe`),
  and a gap on a price-break day (a bad split adjustment) is left out.
- **Universe:** judged at the prior close: $5, top 50% by 50-day dollar volume, no price break in the past
  year. `--size sp500` (or sp400 / sp600 / other) keeps one index.
- **Trade:** buy an up gap, short a down gap. In at the day 0 close (the volume is only known then) or the
  next open, out at the close of day 1, 3, 5, 10, 20, 40 or 60. The pick, set before the first run, is
  `close:20`, each direction on its own.
- **Costs:** a round trip on every trade, plus 5%/yr borrow on the shorts (`ret`), and the same against
  the S&P (`hedged`). `adv20_%` is the share that went 20%+ against the trade before the exit.

```
python gap_followthrough/gap_events.py --start 2010-01-01
python gap_followthrough/gap_events.py --start 2010-01-01 --pick open:20 --sample 10
python gap_followthrough/gap_events.py --start 2010-01-01 --size sp500
python gap_followthrough/gap_events.py --start 2010-01-01 --only move20:0:10
python -m pytest gap_followthrough
```

The run prints a day-by-day profile around the up and down gaps, every entry and hold for each
direction, and the pick split by gap size, volume, S&P index, price, dollar volume, what the stock did
after the open on day 0, its 60-day move before the gap, and the stock and the S&P against their 200-day
averages. Then the pick by 5-year block and by year.

## First Dev run (since 2010) and what was added after it

17,698 gaps, about 516 up and 540 down a year. The pick, in at the day 0 close and out at the close of
day 20, had no edge either way after costs: buying up gaps -0.04% a trade (hedged -0.84%, t -2.6, 7 of 17
years up), shorting down gaps -1.42% (hedged -0.53%, 2 of 17 years up). There was no drift after the day 0
close. The day itself moved on from the open (+1.2% for up gaps, +1.5% for down gaps), but the 3x volume is
only known at the close. Two tables were added, both read as context since the pick failed:

- **Day 0 from the open:** in at the day 0 open, out at the close of day 0, 1, 5 or 20, on every 5%+ gap
  with no volume rule (all known at the open) next to the 3x events (flattered, since their volume is
  only known at the close).
- **Lead-up:** the pick by what the stock did in the 10-30 days to the prior close: the 10-, 20- and 30-day
  move in the gap's direction, the 20-day high-low range, the last 10 days' volume against the usual, and
  the distance from the 52-week high (up gaps) or low (down gaps). These are 34 buckets a side on a base
  with no edge, so a bucket is worth a follow-up only if it turns the sign and holds in every 5-year block.
  `--only move20:0:10` (any lead-up column, lower bound included, upper left out) reruns everything on one
  bucket, which shows it by block and by year.

## Second Dev run: lead-up and day 0 from the open

- **Lead-up:** no group made buying up gaps beat the S&P. The best was up gaps within 2% of the 52-week
  high: +0.89% a trade, +0.38% hedged (t 0.65). Gaps in weak stocks were worst: 25%+ below the 52-week
  high lost 1.42% hedged (t -3.1). On the down side the strongest groups were bounces. Stocks already down
  20%+ in the 20-30 days before a down gap rose afterwards, and the short lost 5.7-5.9% a trade (hedged t
  -2.2 to -2.7). That is the capitulation buy the crash rebound study tested, where it came mostly from
  March 2020 and 2022.
- **Day 0 from the open:**

  | In at the day 0 open, out at its close, after costs | Up gaps (buy) | Down gaps (short) |
  |---|---|---|
  | Every 5%+ gap, known at the open | -0.55% (t -6.6) | -0.62% (t -2.9) |
  | Only the 3x-volume gaps, known at the close | +0.68% (t 6.8) | +0.99% (t 6.4) |

  The gaps that keep going are the ones that end the day on heavy volume, so the 3x row can't be traded
  from the open. Catching it would need intraday volume (the first few minutes), which the daily data
  doesn't have.

Lessons carried over from the earlier studies: short-term edges shrank a lot after 2010 (reversion,
breakouts), a 3-day crash of 30%+ kept falling (crash short, so the down-gap side may overlap it), and
the lock-up short looked strong on Dev but failed the holdout, so the pick is set before the first run
and judged once on the locked half.
