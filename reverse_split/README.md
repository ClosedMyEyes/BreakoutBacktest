# Reverse-split short

**Status: passed its one holdout run (2026-10-15), but small for the account as it stands.** The frozen rule
(1-for-20+ splits, not an index member, short the day 0 close, cover day 20) made +11.56% a trade at 50% a
year borrow on the locked half, t 3.51, and passed all four checks. The earlier-split sizing check failed.
The tail sets the size: one short lost 632%. Sizing to trade: 2% of the account a short and 1% on 1-for-100+
splits, which cut the locked half's worst drop from 40% to 22% (set after the run). See "Holdout result" at
the end.

Idea #2 from the project's `ideas/next_setups.md`. Most reverse splits are forced. A Nasdaq stock that
closes under $1 for 30 business days gets a deficiency notice and about 180 days to fix it, and a reverse
split is the usual fix. Most of these companies burn cash and sell new shares into any strength once the
price is back up (offerings, at-the-market programs, warrant exercises). Do the stocks keep falling after
the split, by enough to pay a high borrow fee?

What the project already saw: the spike short made +11.4% a trade with a reverse split in the past year
(64 trades) against +8.6% without, and the crash short +5.83% (122 trades) against +0.88% for all trades.
Both splits were found after the fact on small samples, so they are a reason to test, not a result.

`reverse_split_events.py` is an event table (no sizing, and no stop on the pick; `--explore` tries stops):

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

## Exploring on Dev (`--explore`)

`--explore` prints only the tables for refining the selection, all on Dev, each row split by 5-year block
(2010-14, 2015-19, 2020-24, 2025+) with `blocks_up`, the number of blocks where it made money. Most trades
are from 2022 on, so a filter that only works there is likely one market, not an edge. Every bucket is
known at the day 0 close:

- **Downtrend:** the move over the 60, 120 and 250 bars before day 0, how far below its 52-week closing
  high, and the stock against its 200-day and 50-day.
- **Market:** the S&P against its 200-day and 50-day, an S&P uptrend (above both and the 50-day rising),
  the S&P's 20-day move, the Nasdaq against its 200-day, and the share of stocks above their 200-day.
- **Volume** (split-consistent shares, turnover / adjusted close): the last 10 days against the usual
  (days -60 to -11), the last 50 days against the 200 before them, and day 0 against the usual.
- **The split and size:** the ratio (with 1-for-20 to 50, 50 to 100 and 100+ apart, since the biggest
  ratios leave the smallest floats), how many reverse splits in the past 3 years (0, 1, 2+), the printed
  price on day -1, the day 0 move, dollar volume and index.
- Every row also shows x3_%, the share whose high reached 3x the entry before the cover, and the worst
  trade, to find where the squeezes come from.
- **Every entry and hold** by block, at the pick's fee.
- **Stops on the pick:** cover when a day's high reaches +300, +200, +100 or +50% above the entry, at the
  stop or at the open when it gapped above it, with the share stopped, the worst trade, the 1-in-20 worst
  and the account test at 2% a short.

`--only` narrows the events first, so the tables show one filter inside another. Each table row is one
more try on the same data and the run prints how many it holds. A filter is worth keeping only if it holds
in every block and has a reason behind it; the rule then gets frozen here before the holdout.

```
python reverse_split/reverse_split_events.py --start 2010-01-01 --explore
python reverse_split/reverse_split_events.py --start 2010-01-01 --explore --pick close0:20
python reverse_split/reverse_split_events.py --start 2010-01-01 --explore --only spx_above200:1:2
python reverse_split/reverse_split_events.py --start 2010-01-01 --explore --only prior_n:1:99 --only ratio:20:inf
```

Flags are 0 or 1, so `col:1:2` keeps the yes side. The explore columns for `--only` are `move120`,
`move250`, `from_high`, `above50`, `vol10_x`, `vol50_x`, `vol0_x`, `spx_above50`, `spx_trend`, `spx_move20`,
`comp_above200`, `breadth` and `prior_n`, on top of `ratio`, `px_prev`, `px0`, `dv_m`, `run60`, `day0`,
`above200` and `spx_above200`.

## The frozen rule (set 2026-10-15, after exploring Dev, before the holdout)

**Rule (`--frozen` = `--pick close0:20 --only ratio:20:inf --only in_index:0:1`):** a reverse split of
1-for-20 or more, by a stock that is not in the S&P 500, 400 or 600 or the Russell 2000 on day 0. Short at
the day 0 close and cover at the close of day 20, no stop. Universe as before: $1 printed, top 80% by
dollar volume, first reverse split in 60 bars. Judged at **50% a year borrow** with the same four checks as
the first pass bar (ret above 0 with t above 2; positive in more than half the years and in 2015-19, 2020-24
and 2025+; about 5% a trade; still positive without March 2020 to December 2021).

**Sizing check, set the same day:** trades with an earlier reverse split in the past 3 years (on all 329
Dev trades in the 1-for-20+ group: 140 at +26.0% a trade against 189 at +6.9%) should beat the rest on the
holdout by more than 0 with a Welch t above 2. If they do, those trades get 2x the size; if not, every
trade is the same size. The run prints it under the pass bar.

`--frozen` prints only this rule: no other entries or holds and no day-by-day table, so the holdout's days
after the cover stay unseen. The day 20 to 80 short in the same group (+14.6% on Dev) is its own setup,
to be explored on Dev later. The filter table (below) prints as context.

Why this rule, from Dev (314 tries by then: 132 in the first run and 91 in each explore run):

- 1-for-20+ was positive in every block in the first explore run (+19.11% a trade at day 60, 316 trades).
  A printed price under $0.50 before the split was too, and includes almost all of these stocks.
- Inside it, close0:20 is +15.06% a trade at 50% a year (329 trades, t 5.79, every block positive,
  win 74%), breaks even at 243% a year borrow, and its worst trade is -257% (against -1420% for the
  first pick). The account at 2% a short had a worst drop of 11.5%. Longer holds made a few points more
  with a bigger tail and a lower t.
- Index members lose: 22 Russell 2000 trades at -25.7% and one S&P trade at -156.6% inside the group, and
  in the first run (all splits, day 60) Russell 2000 -16.0% and S&P -13.0%. They are bigger companies,
  more likely to turn around or be bought. Without them the group is 306 trades at +18.55% (t 7.80, every
  block positive, win 76%, break-even fee 286% a year).
- Every stop made it worse (a +300% stop cut it to +9.70%), so there is none.
- Other filters looked strong inside the group (an earlier reverse split: +23.7 to +30.9% against +6.9%
  for none; 90%+ below the 52-week high: +21.1% against about +2%; $1M+ a day: about +18% against +6.9%),
  but they overlap, each cuts the trades (an earlier split leaves 140 of 329), and they were found on the
  same data. They are left out of the rule and print as context in the filter table, which shows for
  each one how many trades it keeps, their average, the average of the ones it drops, and the total
  (trades x average). On Dev only two raised the total: not an index member (now in the rule) and not
  after the S&P rose 5%+ in 20 days, both by dropping a few big losers.
- Only 7 of the Dev trades are from 2010-14 and about 40 from 2015-19, so the edge is mostly 2020 on.

The holdout is the other half of the stocks (split by symbol and sector) over the same years, so it should
have about as many trades. It tests whether the Dev result was luck in which stocks landed in Dev, not
whether 2022-26 was a special market for these shorts.

`python reverse_split/reverse_split_events.py --start 2010-01-01 --frozen` prints the same thing on Dev,
including the exact Dev filter table. The holdout run, once, with no changes after it:

```
python reverse_split/reverse_split_events.py --start 2010-01-01 --frozen --universe holdout --unlock-holdout
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

## Holdout result

Run once on the locked half on 2026-10-15 (`--frozen --universe holdout --unlock-holdout`). **Passed all four
checks.** No changes after it.

| | Dev | Locked half |
|---|---|---|
| Trades (a year) | 306 (18.4) | 263 (15.8) |
| Short, % a trade at 50%/yr borrow (t) | +18.55 (7.80) | +11.56 (3.51) |
| Hedged with the S&P | +19.69 | +12.99 |
| Win rate | 76% | 70% |
| Break-even borrow, % a year | 286 | 197 |
| Without March 2020 to December 2021 | | +12.26 (t 3.66) |

1. ret above 0 with t above 2: +11.56%, t 3.51. Yes.
2. Positive in 10 of 13 years, and in 2015-19 (+7.68, 18 trades), 2020-24 (+14.83, 131) and 2025+ (+8.77,
   110). Yes. 2010-14 had 4 trades.
3. +11.56% a trade, 16 a year. Yes.
4. +12.26% without March 2020 to December 2021. Yes.

At a 1.5% round trip it is +10.56%. Six trades (2.3%) stopped trading during the hold, averaging +39.1%.

**Sizing check: failed.** Trades with an earlier reverse split made +10.71% (89 trades) against +11.99% for
the rest (174), difference -1.28, t -0.14. Every trade is the same size.

**The filter table on the locked half** (context, set before the run, not tested):

| Keep only | Trades | Avg | Dropped avg |
|---|---|---|---|
| All trades | 263 | +11.56 | |
| Down 90%+ over 250 days | 145 | +17.14 | +4.70 |
| 90%+ below the 52-week high | 162 | +15.40 | +5.38 |
| $1M+ a day traded | 172 | +14.71 | +5.59 |
| Down 75%+ over 120 days | 132 | +13.26 | +9.83 |
| Volume last 50 days 1.5x+ the 200 before | 187 | +12.05 | +10.35 |
| S&P above its 200-day | 216 | +11.38 | +12.35 |
| An earlier reverse split in 3 years | 89 | +10.71 | +11.99 |
| Not after the S&P rose 5%+ in 20 days | 222 | +9.75 | +21.32 |
| Day 0 volume 2x+ usual | 158 | +9.30 | +14.95 |

The fast-collapse filters and $1M+ a day did better on both halves. The earlier split, the S&P filters and
the volume filters did not hold up. Any of them would have to prove itself on trades from here on.

**The tail sets the size.** The worst trade, DFNS (day 0 2026-07-20), ran to 20x the entry at its peak and
lost 632% at the cover; the next were EDVA -281% and RSLS -185% (peak 11x). The account at 2% a short grew
3.6% a year with a worst drop of 39.6% (June to July 2026). At 5% it grew 8.6% a year but dropped 98.5%. On
average 1.25 shorts were open (at most 11), so little of the account is used. At 1% a short, a DFNS costs
about 6% of the account at the cover and about 20% at its peak, and the account grows about 2% a year.

**Half size on 1-for-100+ (added after the holdout run).** Three of the five biggest spikes on the locked
half were 1-for-70 or more. On Dev, 1-for-100+ splits reached 3x the entry most often (7.7% of 39 trades,
against 3.4% of 208 for 1-for-20 to 50) but lost no more by the cover (+18.91% a trade). So the account
table now also shows each size with the 1-for-100+ shorts at half size (`size_100+_%`). It was picked after
seeing DFNS (1-for-125), so the locked half is not a clean test of it.

| Account at 50%/yr borrow | Growth a year | Worst drop | When |
|---|---|---|---|
| Dev, 2% a short | 6.95% | -10.9% | Oct 2023 to Apr 2024 |
| Dev, 2%, 1-for-100+ at 1% (39 of 306 trades) | 6.50% | -11.9% | Oct 2023 to Jan 2024 |
| Locked half, 2% a short | 3.61% | -39.6% | Jun to Jul 2026 (DFNS) |
| Locked half, 2%, 1-for-100+ at 1% (41 of 263 trades) | 3.43% | -22.0% | Sep 2016 to Jan 2017 (RSLS) |
| Locked half, 5%, 1-for-100+ at 2.5% | 8.44% | -53.4% | Sep 2016 to Jan 2017 |

On the locked half it nearly halves the worst drop for 0.2% a year. On Dev it costs 0.45% a year and
doesn't help the drop, since Dev had no DFNS. **Sizing to trade: 2% of the account a short, 1% on 1-for-100+,
expecting a drop of about 22%.** The worst drop left is RSLS (1-for-70). Lowering the cut to 1-for-50 would
catch it, but 1-for-50 to 100 was the best Dev bucket (+23.85%) and it would be tuning on the locked half.

DFNS's 13D (a 16.7% holder, filed after the close on July 24, the Friday before the run) was not a
purchase: the company issued those shares to pay for 60% of a drone maker, a deal announced on July 10,
before the split. Covering on a 13D would have saved this trade, but filings aren't in the data, so it
is untested.
