# Reverse-split short

**Status: locked on 2026-10-15 and worth trading. See "How to trade it" at the end.** The frozen rule
(1-for-20+ splits, not an index member, short the day 0 close, cover day 20) made +11.56% a trade at 50% a
year borrow on the locked half, t 3.51, and passed all four checks. The earlier-split sizing check failed.
The tail sets the size: one short lost 632%. Sizing to trade: 2% of the account a short and 1% on 1-for-100+
splits, which cut the locked half's worst drop from 40% to 22% (set after the run). See "Holdout result" at
the end. **The day 20 to 80 short passed its own holdout run too** (+15.15% a trade, t 2.29, 5 a year, about
half its Dev result) **and was locked the same day at the same size;** its checklist is under "How to trade
it". A code audit found no look-ahead. With real costs, the trades still open marked and the short at the next
open, both shorts made +11.41% a trade on the locked half (t 3.79) against +12.43% as run (see "Audit" at
the end).

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
known at the day 0 close, or by the entry close for a later entry:

- **At the entry** (only with a `close5` or `close20` pick, such as the day 20 to 80 short `close20:60`):
  the move since the day 0 close, the highest high since then and how far the entry close is below it, the
  printed price (under $1 is hard to short), dollar volume, the volume since day 0 against before the split,
  and the S&P against its 200-day, all at the entry close.

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
python reverse_split/reverse_split_events.py --start 2010-01-01 --explore --pick close20:60 --only ratio:20:inf --only in_index:0:1
```

Flags are 0 or 1, so `col:1:2` keeps the yes side. The explore columns for `--only` are `move120`,
`move250`, `from_high`, `above50`, `vol10_x`, `vol50_x`, `vol0_x`, `spx_above50`, `spx_trend`, `spx_move20`,
`comp_above200`, `breadth` and `prior_n`, on top of `ratio`, `px_prev`, `px0`, `dv_m`, `run60`, `day0`,
`above200` and `spx_above200`. With a `close5` or `close20` pick, the entry columns are `since0`,
`peak_since0`, `off_peak`, `px_e`, `dv_e_m`, `vol_since_x` and `spx_above200_e`. With one of those in
`--only`, the every-entry-and-hold table leaves out the entries before the pick's, since they would be
picked with what happened after them.

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
frozen later as `--frozen 20-80` (its section is at the end). The filter table (below) prints as context.

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

## How to trade it (locked 2026-10-15)

**Finding the splits ahead of time.** A reverse split is public before day 0. First comes a shareholder
vote on a range of ratios (a proxy, weeks or months ahead). Then the board sets the ratio and the date in a
press release and an 8-K, usually 2 to 10 days ahead. Nasdaq has required this at least 2 business days
before the split takes effect since late 2023. Places to watch: split calendars (Benzinga, TipRanks),
Nasdaq's Daily List, the broker's corporate actions page, or an SEC full-text search for 8-Ks that mention
"reverse stock split".

**On day 0 (the first day at the new price), before the close:**

1. The final ratio is 1-for-20 or more. Check it on the day: boards change it late (DFNS went from 1-for-50
   to 1-for-125 the week before).
2. Not in the S&P 500, 400 or 600 or the Russell 2000.
3. Printed price $1 or more, and not in the bottom 20% of stocks by 50-day dollar volume.
4. It is the stock's first reverse split in 60 trading days.
5. Shares can be borrowed. Check that morning, since many of these can't be. The test charged 50% a year;
   it breaks even at about 200%.
6. Short at the close: 2% of the account, 1% for 1-for-100+. If the stock fell 10%+ that day, the short-sale
   rule is on and a market-on-close short may not fill. The close matters: a short at the next open instead
   made about a point a trade less (Audit).
7. Cover at the close 20 trading days later. No stop: every stop tested made it worse.

What to expect, from the locked half: about 16 trades a year, about 1.3 shorts open at once, +11.6% a trade
after borrow (+10.8% with real spreads and commission), 3.4% a year for the account at this size and a worst
drop of about 22%. A 13D or a big deal while short is worth a look by hand, but covering on one is untested.

**On day 20 (the day 20 to 80 short, locked 2026-10-15), before the close:**

1. The split passed the day 0 checks above (steps 1 to 4), and the stock traded $1M+ a day on average over
   the 50 days to day 0.
2. The close is 25% or more below the highest high since day 0 (days 1 to 20). A stock still near its
   post-split high is a squeeze still running: those lost money.
3. Printed price $1 or more.
4. The S&P 500 closes above its 200-day average that day.
5. Shares can be borrowed. It breaks even at about 115% a year on the locked half.
6. Short at the close: 2% of the account, 1% for 1-for-100+. When the stock is already short from day 0,
   keep it and bring it back to that size, which saves a round trip. If a check is too close to call before
   the close, a short at the next open did as well in the audit (+15.7% a trade against +15.2%).
7. Cover at the close 60 trading days later (day 80). No stop.

What to expect, from the locked half: about 5 trades a year, +15% a trade after borrow (the same with real
costs and the trades still open marked), about 2.4% a year
for the account at this size and a worst drop of about 26%. That drop was one stock, FLZH, whose high went
14x the entry before it fell back. At 2%, a spike like that marks the short at about a quarter of the
account, so keep enough margin to sit through it, or the broker may buy the short in at the top.

**Both shorts together** (`--combined`, run on both halves on 2026-10-15). Every 20-80 short followed a 0-20
short on the same split (136 of 136 on Dev, 84 of 84 on the locked half). At 2% a short, 1% on 1-for-100+:

| | Dev | Locked half |
|---|---|---|
| 0-20 alone | 6.50%/yr, worst drop 11.9% | 3.43%/yr, worst drop 22.0% |
| 20-80 alone | 4.74%/yr, worst drop 17.2% | 2.43%/yr, worst drop 26.5% |
| Both | 10.87%/yr, worst drop 16.2% | 4.66%/yr, worst drop 25.6% |
| Both, each year 2022 to 2026 | +13.7, +37.0, +34.2, +56.7, +28.8 | +11.6, +23.1, +24.1, +22.0, +3.7 |
| Shorts open at once, most / average | 22 / 3.3 | 18 / 2.5 |
| Most of the account short at once | 44% | 55% |

The drops don't stack: the 0-20 shorts were making money while FLZH spiked. The yearly rate over 2010-26
undersells the account now. Almost every trade is from 2022 on, when reverse splits became common, so the
years before sit near zero. If they stay as common as in 2022-26, plan on about 16% a year from both shorts
at 2% (the locked half's 2022 to 2026 average with every fix in the audit below: +10.9, +19.8, +21.2, +21.3
and +6.2 to October; 2023-25 alone is the best stretch, not the plan), with a drop of about 25%. At 5% a
short the locked half would have been 228% short at the peak, so 2% is the size.

## The day 20 to 80 short (frozen 2026-10-15, then run once on the holdout)

**Rule (`--frozen 20-80` = `--pick close20:60 --only ratio:20:inf --only in_index:0:1 --only
off_peak:25:inf --only dv_m:1:inf --only px_e:1:inf --only spx_above200_e:1:2`):** the same splits as the
0-20 rule (1-for-20+, not in the S&P 500, 400 or 600 or the Russell 2000, same universe on day 0) with $1M+
a day traded (50-day) on day 0. At the day 20 close, the stock is 25%+ below its highest high since day 0
and prints $1+, and the S&P is above its 200-day. Short at the day 20 close and cover at the close 60
trading days later (day 80), no stop. Judged at **50% a year borrow** with the same four checks. Sizing as
for the 0-20 short: 2% of the account a short and 1% on 1-for-100+ (the account table prints both).

**S&P check, set the same day as context (not part of the pass bar):** the run also prints the trades that
every other filter keeps and the S&P filter drops. On Dev they were 26 trades at +1.9% a trade, against
+26.5% for the rule. The rule keeps the filter whatever the holdout shows.

Why this rule, from Dev (242 more tries in two explore runs of 121 rows, on top of the 314 before the
first freeze):

- In the 0-20 group, `close20:60` made +15.6% a trade (285 trades, t 3.5), but its worst trade lost 694%
  and the account at 2% a short dropped 32%.
- How far the stock had faded from its post-split high by day 20 split it. Within 10% of the high: -32.2%
  (16 trades); 10-25% below: +3.7% (40); 25-50% below: +13.5% (112, up in 4 of 4 blocks); 50%+ below:
  +28.3% (117, 3 of 3). The stocks still near the high are squeezes still running.
- Under $1M a day traded on day 0: +1.8% (84 trades, up in 1 of 4 blocks), against about +21% for the
  rest. This filter also held on the locked half for the 0-20 short.
- Under $1 at day 20: the 15 trades did best (+38%), but these are the hardest to borrow and short, so the
  rule asks for $1 as on day 0.
- Inside those three (162 trades, +22.5%, t 5.8, up in every block), the S&P at the day 20 close split it:
  above its 200-day +26.5% (136 trades, t 7.1, up in every block), below +1.9% (26 trades, t 0.2, -25.6% in
  2025+). Above beat below in every block that had both. A likely reason: a three-month short that starts
  in a weak market runs into its rebound, and these stocks bounce hardest. The S&P on day 0 split it less
  (above +23.6%, below +15.8%, both up in every block), the worst Dev trade (-225%) is in the below group
  (about +11% without it), and the S&P filters failed on the locked half for the 0-20 short. That is why
  the dropped trades print again on the holdout.
- Every stop made it worse again (+300%: +12.2% a trade), so there is none. A longer hold made a little
  more (`close20:120` +29.1%), but the hold was set at 60 before exploring.
- Only one Dev trade is from 2010-14, and most are from 2022 on.

On Dev the rule is 136 trades at +26.5% a trade (t 7.1, up in every block, win 76%, break-even borrow
166% a year, worst -216%). `--frozen 20-80` prints only this rule, like `--frozen`, so the holdout's other
entries and holds stay unseen. On Dev first, then the holdout once, with no changes after it:

```
python reverse_split/reverse_split_events.py --start 2010-01-01 --frozen 20-80
python reverse_split/reverse_split_events.py --start 2010-01-01 --frozen 20-80 --universe holdout --unlock-holdout
```

### Its holdout result

Run once on the locked half on 2026-10-15 (`--frozen 20-80 --universe holdout --unlock-holdout`), after the
same rule on Dev. **Passed all four checks.** No changes after it.

| | Dev | Locked half |
|---|---|---|
| Trades (a year) | 136 (8.2) | 84 (5.0); 17 more still open when the data ends |
| Short, % a trade at 50%/yr borrow (t) | +26.45 (7.07) | +15.15 (2.29) |
| Hedged with the S&P | +30.03 | +18.29 |
| Win rate | 76% | 77% |
| Went 50%+ against the short first | 33% | 43% |
| Break-even borrow, % a year | 166 | 115 |
| Account at 2% a short, 1% on 1-for-100+ | 4.74%/yr, worst drop 17.2% | 2.43%/yr, worst drop 26.5% |

1. ret above 0 with t above 2: +15.15%, t 2.29. Yes.
2. Positive in 5 of 8 years, and in 2015-19 (+7.06, 2 trades), 2020-24 (+26.88, 39) and 2025+ (+4.89, 43).
   Yes. 2026 was -18.62% on 14 trades.
3. +15.15% a trade, 5 a year. Yes.
4. +16.83% without March 2020 to December 2021 (t 2.46). Yes.

At a 1.5% round trip it is +14.15%. Three trades stopped trading during the hold, averaging +39.2%.

**The S&P check (context):** the 19 trades every other filter kept and the S&P filter dropped made +8.23% a
trade (t 1.17), against +15.15% for the rule. The same direction as on Dev (+1.87%), with a smaller gap.

**The tail.** The worst trades were AGMH (1-for-50, June 2025, -252.7%, its high 611% above the entry),
IPST (1-for-20, April 2026, -210.3%) and NAKA (1-for-40, May 2026, -129.8%). The account's worst drop (26.5%,
March 20 to 25, 2026) looks like FLZH (1-for-25), whose high reached 1,312% above the entry (14x) before it
fell back and was covered at -34.9%. That is inferred from the dates; worth checking on a chart in case it
is a bad print. Half size on 1-for-100+ didn't touch it (FLZH is 1-for-25) and cost 0.46% a year here
(0.10% on Dev). At 2% a short with no half size the account made 2.89% a year; at 5%, 6.85% a year with a
64% drop.

**Context** (not tested, not part of the rule). Better on both halves: not after the S&P rose 5%+ in the
20 days before day 0 (locked half: 67 trades at +21.30% against -9.05% for the 17 it drops; Dev:
+27.88% against +13.98%). An earlier reverse split did worse on the locked half (+10.46% against +18.35%)
and the same on Dev.

**Both shorts in one account.** `--combined` puts the 0-20 and 20-80 shorts in one account (sized to trade:
2%, 1% on 1-for-100+), each alone and together, with the account's return by year. Most 20-80 shorts
follow a 0-20 short on the same split, so in practice the short is held on from day 20 to day 80. It adds no
new rule, so it can run on both halves:

```
python reverse_split/reverse_split_events.py --start 2010-01-01 --combined
python reverse_split/reverse_split_events.py --start 2010-01-01 --combined --universe holdout --unlock-holdout
```

## Audit: look-ahead and shortcuts that could flatter the results (2026-10-15)

After both holdout runs, the code was read end to end for anything that uses information from after the
decision or makes the shorts look better than they could be traded.

**No look-ahead found** in what picks the trades:

- Index membership (S&P 500/400/600, Russell 2000) and the major-exchange flag are Norgate's day-by-day
  history, so day 0 uses the membership on day 0. Delisted stocks are in the data.
- The split is found from the change in Norgate's adjustment factor between day -1 and day 0, which only
  moves with an event on day 0. The ratio filter uses that day's jump. The $1 floor uses the printed price.
- Liquidity (50-day $ volume and its rank across listed stocks), the 252-bar price-break block (judged at
  day -1), the pinned-stock check, the dedupe (earlier splits only) and the earlier-split count all use bars
  up to day 0. The S&P's 200-day is a trailing average of its closes.
- The day 20 filters (25%+ below the high since day 0, $1+ printed, the S&P) use bars up to the day 20
  close, and the $1M+ a day filter uses day 0.
- The account sizes each short on the account's value at its entry and marks every open short at each close.
- The Dev and holdout halves are split by company with a fixed seed, before any results were seen.

One step does read later bars: a factor jump undone within 5 bars counts as a bad print, not a split. It
left out 0 splits in every run so far. Two things work against the shorts and stay as they are: a hold
across a downward price break is left out (an untrusted gain for a short), and a stock that stops trading is
covered at its last exchange close.

**What could flatter them**, and `--audit` measures each one on the two frozen rules, each alone and both in
one account at 2% a short (1-for-100+ at 1%):

- **Trades still open when the data ends are left out.** 15 of the 99 day 20 to 80 shorts on the locked
  half were (all from mid-2026, when that rule was losing). `open marked` covers them at the last close.
- **Costs.** The 0.50% round trip charges commission as if the stock were $10 and 0.2% slippage a side.
  `real costs` charges $0.005 a share at the stock's own printed price, 0.5% slippage a side, the ordinary
  dividends going ex while short, and the borrow for every market day held, not just the stock's bars.
- **The close the rule is decided on is the close it trades.** Both rules are checked a few minutes before
  the close, so a borderline stock can pass at 3:50pm and fail at the close, or the reverse. `next open`
  shorts at the next open instead, with the same cover: the worst case for that.
- **Planning numbers.** The Dev results were picked from many tries, so the holdout's are the ones to plan
  on. The 1-for-100+ half size was picked after the 0-20 holdout run.

```
python reverse_split/reverse_split_events.py --start 2010-01-01 --audit
python reverse_split/reverse_split_events.py --start 2010-01-01 --audit --universe holdout --unlock-holdout
```

It changes no rule, so it can run on both halves. The table has trades, the average short at 50%/yr and at
100%/yr, t, win rate, and the account's growth and worst drop for each rule and each fix, then the account's
years as run and with the fixes, then the counts behind them (the trades still open and what they average,
the trades left out for a downward break, the entries off the exchanges, the printed price and the costs).

### Its result

Mark ran `--audit` on both halves on 2026-10-15. Short, % a trade at 50%/yr borrow (t), and the account at
2% a short (1% on 1-for-100+):

| | As run | Open trades marked | Real costs | Next open | All three |
|---|---|---|---|---|---|
| Locked half, 0-20 | +11.56 (3.51) | +11.58 | +10.80 | +10.52 | +9.79 (2.98) |
| Locked half, 20-80 | +15.15 (2.29) | +16.01 | +14.29 | +15.72 | +15.79 (2.88) |
| Locked half, both | +12.43 (3.77) | +12.78 | +11.64 | +11.78 | +11.41 (3.79) |
| Locked half, both, account | 4.66%/yr, drop 25.6% | 5.06% | 4.35% | 4.43% | 4.52%, drop 25.0% |
| Dev, 0-20 | +18.55 (7.80) | +19.02 | +17.79 | +17.42 | +17.09 (6.94) |
| Dev, 20-80 | +26.45 (7.07) | +25.79 | +25.58 | +26.02 | +24.58 (6.99) |
| Dev, both, account | 10.87%/yr, drop 16.2% | 11.38% | 10.45% | 10.29% | 10.35%, drop 16.5% |

- **The edge holds.** With every fix, both shorts made +11.41% a trade on the locked half, 1.0 point under
  as run, with the same t. The 0-20 short loses about 1.8 points (+9.79%, t 2.98) and the 20-80 short none.
- **The trades left out at the data end were winners**, so leaving them out was conservative: on the locked
  half 4 for 0-20 (+13.47% marked at the last close) and 15 for 20-80 (+20.80%); on Dev 5 (+48.02%) and 12
  (+18.35%).
- **Costs are the real hit.** The median entry prints $4.37 (0-20) and $2.92 (20-80) on the locked half, so
  commission at $0.005 a share and 0.5% slippage a side come to 1.23% and 1.34% a round trip, about 0.8
  points a trade more than the 0.50% the rules were tested at. No trade paid a dividend.
- **The next open** costs the 0-20 short about a point a trade (the split-day close is the better price) and
  helps the 20-80 short a little on the locked half (+15.72% against +15.15%; on Dev -0.4).
- **Counts.** 0 splits left out by the bad-print filter on either half; 0 and 2 trades left out for a
  downward break (Dev, locked half, 20-80); 0 entries off the major exchanges. One symbol listed after the
  Dev/holdout split file was made is in neither half.
- **Both shorts' account by year on the locked half with every fix:** 2022 +10.9%, 2023 +19.8%, 2024
  +21.2%, 2025 +21.3%, 2026 +6.2% to October (+3.7% as run, +8.8% with the open trades marked and real
  costs). The planning case is now about 16% a year at 2% with a drop of about 25%.
