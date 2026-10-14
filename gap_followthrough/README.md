# Big-gap follow-through

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
python -m pytest gap_followthrough
```

The run prints a day-by-day profile around the up and down gaps, every entry and hold for each
direction, and the pick split by gap size, volume, S&P index, price, dollar volume, what the stock did
after the open on day 0, its 60-day move before the gap, and the stock and the S&P against their 200-day
averages. Then the pick by 5-year block and by year.

Lessons carried over from the earlier studies: short-term edges shrank a lot after 2010 (reversion,
breakouts), a 3-day crash of 30%+ kept falling (crash short, so the down-gap side may overlap it), and
the lock-up short looked strong on Dev but failed the holdout, so the pick is set before the first run
and judged once on the locked half.
