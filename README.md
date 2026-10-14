# BreakoutBacktest

Backtest and (later) nightly scanner for the O'Neil / Minervini long-side Trend Template
system, and a separate short side (see "Short side" below). It follows the build plan in `Trend_Template_System_Plan.docx` (project files):
phases, ground rules and decisions are referenced by number in the code.

This is a separate program from Stockbot and never runs inside the live trading process.

## Status

| Phase | What | State |
|---|---|---|
| 1 | `pipeline/build_data.py`: Norgate → local Parquet, sanity checks | Written; tested on synthetic data |
| 2 | `pipeline/build_features.py`: indicators, RS ranks, market features, forward labels | Written; tested |
| 3 | `pipeline/make_split.py`: locked Dev/Holdout split; walk-forward windows in `breakout/grid.py` | Written; tested |
| 4 | `core/engine.py`, `breakout/setup_trend.py`, `breakout/grid.py`: fill model, exits, Modes B and C | Written; tested |
| 5 | `breakout/filter_test.py`: Mode A | Written; tested |
| 5.4 | Kill test: F1 + X2 vs date-matched F0 | `breakout/grid.py` is set up for it |
| 6 | Entry families | F0–F6 done. F4 = zigzag swing points (3% or 5% swings, `--vcp-z`), F5 = pole + flag, F6 = first retest of an F1–F5 breakout, bought at the next open |
| 7 | Exit models | X1, X2 (original or breakeven stop after the +20% hold), X3, X4 (sell part at +2R or +X%, stop to breakeven, trail the rest), X5, X6 and time exit done |
| 8 | Mode C portfolio | Done, with Monte Carlo drawdown range |
| 9–11 | Market light, fundamentals, earnings dates | Market features computed; rules to do |
| 12–14 | Holdout run, scanner, forward test | Spike short: holdout passed (+5.45%/trade, t 2.41), `spike_short/spike_scan.py` scanner and forward-test log done |

The long side is parked (best version: `--family f3 --stop-mode structural --market above200`).

## Short side

Follows the Short Side Plan doc. It is kept apart from the long side: its own rules (`rules_short.py`),
engine (`engine_short.py`), scripts (`setup_short.py`, `filter_short.py`) and defaults, so no long result
moves. The long scripts never import the short files.

| Setup | Short when | Stop |
|---|---|---|
| S1 simple breakdown | close below the N-day low (`--n 20/50`) | N-day high |
| S2 bear flag | drop of D% in W weeks, light-volume bounce of 5+ bars taking back at most R%; close below the flag low | flag high |
| S3 failed leader | a former leader (passed the long template in the past year) closes below a falling 50-day on V x volume; short that close | 10-day high |
| S4 rally into resistance | high within X% of a falling 50-day / 21-day EMA, closes below it and down; short that close | that day's high |
| S5 quiet breakdown | the F3 mirror: ATR10 <= r x ATR50, vol10 <= v x vol50; close below the 10-day low | 10-day high |
| S6 impulse retest failure | the daily version of Stockbot's impulse short: a breakout to a peak, a pullback of 60-95% of the move, a rally back to within 3% of the peak and a red day that fails under it; short that close | peak + 10% of the move |
| S7 slow fade | a top within 5% under the 52-week high (not broken), then a quiet drift 3-15% down; close below the fade's low | 10-day high |
| S0 | random stocks from the same pool on the same days, same time of day, risk % and exits | |

- Pool: printed price $10+, top 30% by dollar volume, the downtrend template (Minervini's eight rules
  mirrored, RS 30 or lower; `--former-leader on` adds the leader rule), and the S&P 500 below its 200-day
  (`--market below200`, the default; `downtrend` = also below a falling 50-day; `off`). S3, S6 and S7 short stocks that were strong
  until recently, which rarely pass the template yet, so they skip it (S3 adds the former-leader rule);
  `--pool on` adds the template, `--pool off` drops it for any setup.
- Exit XS2: stop 7.5% above the entry (or `--stop-mode structural`, skipped if over 15%), cover at 20%
  profit; down 20% within 3 weeks switches the target off and covers after a close above the 21-day EMA;
  cover after 60 bars at most.
- Costs: slippage and commission as on the long side, borrow 0.5% a year (`--borrow-pct 3` to test),
  and every dividend going ex while short (the `div` column from Norgate's total-return gap).
- Price breaks are flipped: a downward break is the gain a short can't trust (covers at the prior close),
  an upward one counts as a loss (`--up-break exit` to treat it like a downward one).

```
python breakout/filter_short.py --breakdowns  # Mode A: do downtrend-template stocks keep falling?
python breakout/setup_short.py                # S1 vs S0, S&P below its 200-day
python breakout/setup_short.py --family s5 --market off
python breakout/setup_short.py --family s3 --sample 10 --sample-result loss
```

Short-side Mode C (portfolio) is not built yet.

## Stretch study (mean reversion)

Both breakout sides stopped working after 2010, so the next question is whether short-term mean
reversion exists on the daily chart. `reversion_test.py` ranks every stock each day by how stretched
it is (3-day move in ATR units, RSI(2), close position in the day's range), splits them into buckets
and prints each bucket's next 1/3/5/10-day return against the S&P 500, with tables by 5-year block,
S&P above/below its 200-day, the stock above/below its own 200-day, and size. No setups or costs yet.

```
python reversion/reversion_test.py                # all years, Dev
python reversion/reversion_test.py --start 2010-01-01
python reversion/reversion_test.py --measure ibs  # breakdown tables for another measure
python reversion/reversion_test.py --start 2010-01-01 --spike-min 30
```

It ends with a spike table for the short side: the first day a stock is up 10/20/30/50%+ over 3 days,
split by the stock and the S&P being above or below their 200-days, with the raw 5/10/20-day move a
plain short pays and the 10-day short after costs, then by 5-year block for one threshold.
After that, a gap-up blow-off table (a small-cap day trader's rule): after a run of 30/50/100%+ from the
10-day low, short the first day that opens above the prior day's high, at its open or close, or only if
it closes red, next to shorting the run's first day with no gap; then by year for `--spike-min`.

`spike_short.py` trades those spike shorts through the short engine with stops and every cost
(slippage, commission, borrow, dividends): runs measured over 3 or 10 bars, 50/100%+, shorted at the
close or at the next gap-up open, stops none/15/25/40% above the short, covered after `--hold` bars.
Prints a grid (win rate, average, the 1-in-20 loss, worst, % stopped, a 50%/yr borrow stress, total,
drawdown, years up) and a by-year table for `--pick K:X:entry:stop`.

```
python spike_short/spike_short.py --start 2010-01-01
python spike_short/spike_short.py --start 2010-01-01 --pick 10:100:gap:25 --sample 10
python spike_short/spike_short.py --start 2010-01-01 --pick 10:100:gap:25 --min-gap 5
python spike_short/spike_short.py --start 2010-01-01 --min-gap 2 --stops 25,40 --market above200 --hold 5
```

The pick also gets a squeeze table: its trades split by what was known at entry (gap size, run size,
volume vs normal, RSI(14), at a 52-week high, share price, the stock's and the S&P's 200-day, liquidity,
a reverse split in the past year). `--market above200` only takes trades while the S&P is above its 200-day.
`--sample` prices are split-adjusted to today like TradingView, so a stock that later reverse split can
show a silly price; `real_px` is the price it actually traded at that day (the $5 floor uses that price).

`--sizes 2.5,5,10` adds an account test for the pick: each short is that % of the account when it opens
(sized on the prior close), at most `--max-shorts` (10) open at once, marked to market at every close.
It prints yearly growth, the worst drawdown, the worst year, the most of the account short at once and the
drawdown if the same trades came in a random order, then the return by year. Every run adds a row to the
research log.

```
python spike_short/spike_short.py --start 2010-01-01 --min-gap 2 --mins 100 --stops 25 --pick 10:100:gap:25 --sizes 2.5,5,10
```

Stress the costs: `--slippage-pct 1` (per fill, default 0.2), `--stop-slip 5` (intraday stops fill 5% above
the stop, as when a squeeze or a halt runs through it) and `--borrow-pct 100` (% a year, default 0.5). The pick
also prints how many setups were skipped (a junk bar on the trigger or entry day, or a trade already open).

### Gap-up short: frozen rule and nightly scanner

Frozen on 2026-10-12 and checked once on the locked half of the stocks (2010-2026): a stock whose close
is 100%+ above the lowest close of the 10 bars before it, that opens the next day at least 2% above that
day's high, is shorted at that open with a stop 25% above the short and covered at the close of the 10th
trading day. Dev half +9.25% a trade (296 trades, t 4.85); locked half +5.45% (310 trades, t 2.41, 12/17
years up). Nearly all the gains are from 2018 on in both halves. The locked half is now used for this
family, so changes to it (a bigger minimum run, other exits) can only be judged on trades from here on.

`spike_scan.py` runs the same code on the latest data: tonight's watch list with the open price that
triggers the short, and the setups that triggered in the last 15 trading days with their stop, cover day and
status. Both go into `spike_signals.csv` in the work folder, the forward-test record.

```
python pipeline/build_data.py
python pipeline/build_features.py
python spike_short/spike_scan.py
python spike_short/spike_scan.py --asof 2025-05-07  # what it would have said that evening (not saved)
```

`tradingview/spike_signal.pine` is the same rule as a TradingView indicator, with the steps for a nightly screen
in `tradingview/README.md`.

## S&P index changes

The first event study after the spike short. Index funds must buy every stock added to the S&P
500/400/600 and sell every stock that leaves, at the close before the change takes effect. If that
forced trading overshoots, adds should lag the S&P afterwards (short them) and drops should beat it
(buy them). It lives in its own folder, `index_changes/`, apart from the spike short.
`index_changes/index_events.py` finds every change in the daily index flags (adds, drops and moves
between indexes, buyouts left out) and prints the stock's return against the S&P before the change,
on the flow day, and 1/5/10/20 days after the flow-day close, by transition, by 5-year block and by
year. No stops or exits yet; no rebuild needed.

```
python index_changes/index_events.py                          # every year, Dev
python index_changes/index_events.py --start 2010-01-01
python index_changes/index_events.py --focus "600->none"      # by-year table for one transition
```

## Crash rebound

Idea #5, the mirror of the spike short: after a stock falls 20/30/50%+ in 1 or 3 days, does it bounce?
It lives in its own folder, `crash_rebound/`. `crash_rebound/crash_events.py` takes the first day of each
fall (one event per crash), keeps the stocks that pass the universe filter ($5, top 80% liquidity) and
prints the raw 1-20 day return of a long from that close (or the next open), after costs, then splits one
pick by what was known at entry (200-day, S&P trend, index member or just dropped from one, gap, close in
the range, volume, prior move, price), by 5-year block and by year. No stops or exits yet.
It also has a gap-down capitulation table, the spike short's gap rule flipped: after a stock closes
30/50/70%+ under its 10-bar high, buy the first open below the prior day's low (`--min-gap`), split by gap
size, run size, S&P trend and price, and by year for `--gap-pick`.

```
python crash_rebound/crash_events.py --start 2010-01-01
python crash_rebound/crash_events.py --start 2010-01-01 --pick 1:20 --sample 10
python crash_rebound/crash_events.py --start 2010-01-01 --entry next_open
python crash_rebound/crash_events.py --start 2010-01-01 --gap-pick 50 --min-gap 2
```

## Crash-keeps-falling short

A lead from the crash rebound study: crashes kept falling when the stock had run up first. It lives in
its own folder, `crash_short/`. `crash_short/crash_short.py` shorts the first day of a 3-day 30%+ fall after
a 20%+ rise over the 60 days before (the rule stated before the run), covers after 10 days, and prints the
short's result after costs and borrow (5%/yr, and 100%/yr as a stress), the squeezes (p95, share whose high
went 25%+ above the entry), a grid of fall size x prior move, splits, 5-year blocks and years.

```
python crash_short/crash_short.py --start 2010-01-01
python crash_short/crash_short.py --start 2010-01-01 --entry next_open
python crash_short/crash_short.py --start 2010-01-01 --sample 10
```

`crash_short/crash_trades.py` trades the same rule through the short engine, as `spike_short.py` does:
a grid of close / next-open / bounce entries and stops of none / 25 / 40 / 60 / 100% intraday or 25 / 40 / 60% judged on the close (`25c`:
cover the next open after a close 25%+ above the short). Slippage, commission, borrow and dividends are
charged, with 100%/yr borrow as a stress. Four filters with a reason behind them are flags and are
also stacked one by one, then left out one at a time, in a filter table: no spike-short setup in the
20 bars before (`--skip-spike 20`), crash-day volume 2x+ usual (`--min-vol-x 2`), the S&P above its
200-day (`--market above200`), and 200 days of history before the fall (`--need-200d`). It adds 5-year
blocks, years, exit reasons, a squeeze table (including volume build-up before the fall and distance
from the 52-week low), and an account test (`--sizes`) that runs the crash short alone, the live spike
short alone and both in one account. The holdout pass bar is printed at the end of every run.

The bounce entry shorts a relief rally into a resistance level within 5 bars of the crash. The level is
set with `--bounce-level`: a % above the crash close (`10`), a share of the 3-day fall won back (`r38`),
or the close before the crash day (`prev`). `--bounce-fill` sets the fill: `open` shorts only an open at
or above the level, and `touch` is a limit order resting at the level. With a bounce pick, a table
compares levels 5 / 10 / 15 / 20 / r25 / r38 / r50 / prev with both fills, and the squeeze table adds the
short price vs the crash close and the number of bars after the crash.

```
python crash_short/crash_trades.py --start 2010-01-01
python crash_short/crash_trades.py --start 2010-01-01 --pick next_open:40c --sample 10
python crash_short/crash_trades.py --start 2010-01-01 --skip-spike 20 --min-vol-x 2 --market above200 --need-200d
python crash_short/crash_trades.py --start 2010-01-01 --pick bounce:25c --sizes 2.5,5,10
python crash_short/crash_trades.py --start 2010-01-01 --pick bounce:25c --bounce-level r38 --bounce-fill touch
```

## Crash bounce short

The crash short's bounce entry, split into its own setup in its own folder, `crash_bounce/`, because it
made the most a trade on Dev. `crash_bounce/bounce_short.py` takes the same crash event and shorts the
relief rally into a line known at the crash close. `--pick LEVEL:FILL:WINDOW:STOP` sets the trade:

- level: `10` / `20` (% above the crash close), `r25` / `r38` / `r50` (that share of the 3-day fall won
  back), or `prev` (the close before the crash day)
- fill: `open` (only an open at or above the line) or `touch` (a limit order resting at the line)
- window: the bars after the crash the line has to be reached in (1 = the next day only)
- stop: `0` (none), `25` (intraday), `25c` (judged on the close), or `pre` (cover the next open after a
  close back at or above the close the 3-day fall started from)

`--target` adds a buy limit below the short, as a multiple of the distance from the short price down to
the crash low (the lowest low of the 3 crash days): `0.5` is half way, `1` is the crash low, `1.5` is half
that distance again below it. If it isn't hit, the trade is covered at the close `--hold` bars after the
entry.

It prints every level x fill x window (`--windows 1,2,5`) at the pick's stop, the pick with each stop
(`--stops`), each target (`--targets 0,0.5,0.9,1,1.5`) with each hold (`--holds 5,10,20`), the filter
table, then the pick by block and year, exit
reasons, the squeeze table and, with `--sizes`, the account test with the live spike short. It shares the
filter flags and the cost settings with `crash_trades.py`. The holdout pass bar is printed at the end of
every run.

```
python crash_bounce/bounce_short.py --start 2010-01-01
python crash_bounce/bounce_short.py --start 2010-01-01 --pick r38:touch:2:0 --sizes 2.5,5,10
python crash_bounce/bounce_short.py --start 2010-01-01 --pick prev:touch:5:pre --sample 10
python crash_bounce/bounce_short.py --start 2010-01-01 --target 0.9 --hold 20 --sizes 5
python crash_bounce/bounce_short.py --start 2010-01-01 --skip-spike 20
```

The frozen rule is `prev:touch:1:0`, hold 10 (holdout passed 2026-10-13).
`crash_bounce/account_report.py` trades it and the live spike short (its frozen rule, without the
judges) through a small account, each alone and both together. It prints the per-trade numbers in % of
the position and in R (1R = 25%, the spike short's stop), the balance from `--capital` (default
$10,000) at each year end, the worst drop, and the time in trades: how often a short is open, how many
at once, and how much of the account is short.

```
python crash_bounce/account_report.py --start 2010-01-01
python crash_bounce/account_report.py --start 2010-01-01 --universe all --unlock-holdout --sizes 5,10
```

`crash_bounce/bounce_scan.py` is the nightly scanner for it: each stock that crashed today with the limit
price to short at in the next session, and the shorts entered in the last 15 trading days with the days
left to the cover. It keeps `bounce_signals.csv` in the work folder as the forward-test record.
`crash_bounce/bounce_signal.pine` is the same rule as a TradingView indicator. `crash_bounce/TRADING.md`
has the evening and next-session steps, how to add the indicator, and a TradingView screen for when the
Norgate data isn't current.

```
python crash_bounce/bounce_scan.py
python crash_bounce/bounce_scan.py --asof 2026-08-07     # what it would have said that evening (not saved)
```

## Setup (Windows machine with Norgate)

```
pip install -r requirements.txt
```

The Norgate Data Updater must be running for `build_data.py`.

## Order of operations

```
python pipeline/build_data.py --list-subtypes    # trial week 1: check which security subtypes are common stock,
                                                 # then set COMMON_STOCK_SUBTYPES in core/config.py
python tools/check_adjust.py ABT 2013-01-02      # spin-offs should not look like crashes (config.PRICE_ADJUST)
python pipeline/build_data.py --limit 200        # quick end-to-end check
python pipeline/build_data.py                    # full pull (prints the Phase 1 sanity checks)
python tools/show_data.py ABPO                   # look at one stock's raw bars and adjustment days
python tools/show_data.py --find lehman          # search names; --delisted lists every delisted stock
python pipeline/build_features.py                # Phase 2
python tools/check_features.py --random 5        # compare against TradingView (Phase 2 done-when)
python pipeline/make_split.py                    # Phase 3, once; commit split.csv
python breakout/filter_test.py --breakdowns      # Mode A baseline
python breakout/setup_trend.py                   # Mode B: F1 50-day breakout + X2, vs F0
python breakout/setup_trend.py --mode portfolio  # Mode C
python breakout/grid.py                          # grid / walk-forward (edit the top of the file)
python core/research_log.py                      # what has been tried so far
```

Phase 4 dummy strategy (buy a 20-day high, exit after 20 days):

```
python breakout/setup_trend.py --template off --n 20 --exit time --time-bars 20 --stop-pct 0 --save-trades results/dummy.csv
python tools/check_features.py --trades results/dummy.csv --n 10
```

Without Norgate, `python pipeline/build_data.py --source synthetic` creates fake data so the whole pipeline can be run
and debugged. Set `BB_DATA_DIR` and `BB_WORK_DIR` to keep test data and logs out of the repo.

## Files

The top folder holds only this README, `requirements.txt` and `split.csv`; every script lives in a
folder. Run each command from the top folder, as written in this README. The data (`store/`),
`split.csv`, `results/`, the research log and the scanners' signal files stay in the top folder.

| Folder | What's in it |
|---|---|
| `core/` | Shared code every script imports: paths and defaults, data loading, rules, the two engines, flags, report tables, the research log |
| `pipeline/` | Norgate to Parquet, features, the Dev/Holdout split |
| `tools/` | Hand checks on the data |
| `breakout/` | The breakout backtests: the long Trend Template, the short S1-S7 setups and the grid runner |
| `reversion/` | The stretch study (mean reversion) |
| `spike_short/` | The live gap-up short and its nightly scanner |
| `tradingview/` | The gap-up short as a TradingView indicator, and how to screen for it |
| `index_changes/` | S&P index changes: adds, drops and moves between indexes, returns vs the S&P |
| `crash_rebound/` | Crash rebound: returns after a big 1 or 3-day fall |
| `crash_short/` | Short crashes that follow a run-up: the event study, then real trades with stops |
| `crash_bounce/` | Short the relief rally after a crash, at a line known at the crash close; its nightly scanner and TradingView indicator |
| `lockup_short/` | Lock-up expiry: new listings around the 180-day lock-up |
| `tests/` | Tests for the shared code, the pipeline and the spike short |

| File | Job |
|---|---|
| `core/config.py` | Paths, Norgate names, defaults (walk-forward lengths, min trades, costs) |
| `core/datastore.py` | Loads only the columns and rows a run needs; holds the Holdout lock |
| `core/rules.py` | Universe filter, Trend Template, entry families. Shared with the future scanner |
| `core/rules_short.py` | Short side: universe, downtrend template, market filter, setups S1-S5 |
| `core/engine.py` | Fill model, exits, F0 baseline, portfolio, metrics |
| `core/engine_short.py` | Short side: fills, exits, borrow and dividends, S0 |
| `core/params.py` | Turns every rule default into a CLI flag so `grid.py` keys map onto them |
| `core/reports.py` | Statistics and tables shared by the long and short scripts |
| `core/research_log.py` | Research log (plan 3.8) |
| `pipeline/data_sources.py` | Norgate source and the synthetic source |
| `pipeline/build_data.py` | Phase 1 |
| `pipeline/build_features.py` | Phase 2. Writes `features`, `ranks`, `labels` (forward returns, Mode A only) and `market_features` |
| `pipeline/make_split.py` | Phase 3 Dev/Holdout split |
| `tools/show_data.py` | Raw-data lookups: one stock's bars and adjustment days (`--around DATE` for one trade), name search, delisted list |
| `tools/check_adjust.py` | One stock's closes around a date under each Norgate adjustment type |
| `tools/check_features.py` | Hand checks against TradingView |
| `breakout/filter_test.py` | Mode A |
| `breakout/setup_trend.py` | Modes B and C |
| `breakout/filter_short.py` | Short side Mode A |
| `breakout/setup_short.py` | Short side Mode B: one setup vs S0 |
| `breakout/grid.py` | Your grid runner, extended: windows, walk-forward report, holdout lock, research log |
| `breakout/check_pinned.py` | Signal-day range of buyout-pinned trades vs the rest, to set a minimum-range filter |
| `breakout/regime_test.py` | Candidate market regime indicators (dispersion, small vs large, IPOs, momentum spread, ...) vs a saved trade log |
| `reversion/reversion_test.py` | Stretch buckets and the spike and gap-up tables that led to the spike short |
| `spike_short/spike_short.py` | Spike / gap-up blow-off shorts with stops, squeeze table, account test |
| `spike_short/spike_scan.py` | Nightly scanner for the frozen gap-up short, plus the forward-test log |

A new script in a folder finds the shared code by adding `core/` to its path before importing it, as
every script here does:

```
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
```

## Guard rails built in

- **No look-ahead.** Features on day t use bars up to t. Forward returns live in a separate file that only
  Mode A may load. A test changes all future bars and checks that no past feature moves.
- **Survivorship.** Delisted stocks stay in; trades in a stock that delists exit at its last exchange close.
- **Exchange-listed runs only.** OTC bars outside a listing are dropped (post-delisting OTC prints are often
  garbage). A stock that leaves the exchanges for 20+ bars and comes back is a new security, `SYMBOL~2`,
  on the same Dev/Holdout side as `SYMBOL`. Data ending 14+ days before the last date counts as delisted.
- **Price breaks.** Where old and new price history don't connect (a bankruptcy exit, de-SPAC or micro-cap
  reverse split where the adjusted close still jumps, or any 10x / -90% bar), the stock gets no new entries
  for 252 bars, because its moving averages and 52-week range span two different securities. A break that
  jumps UP is never booked as a gain: a trade open across it exits at the prior close, and Mode A leaves out
  forward returns that span it. Downward breaks count as real losses, so any error is on the conservative side.
- **Holdout lock.** `holdout` and `all` universes refuse to load without `--unlock-holdout`.
- **Date-matched F0.** Every Mode B run reports `f0_avg_R` and `edge_R`: random template stocks on the
  same signal days, same risk, same exits.
- **Count the tries.** `grid.py` logs the number of combinations to the research log, and the walk-forward
  report shows train vs test so the honest (test) number is always next to the flattering one.
- **Mode A statistics by date** with Newey-West t-stats, so overlapping forward windows are not counted as independent.

## Speed and memory (measured on synthetic data at about 40% of the real size)

| Step | Time | Peak RAM |
|---|---|---|
| Feature build (22.8M rows) | 3 min, plus about 2 min for the F4 swing points | — |
| Mode A, full history | 14 s | 2.2 GB |
| Mode B with 3 F0 seeds, full history | 14 s | 2.8 GB |
| Mode B, 10-year window | 8 s | 2.3 GB |
| Mode C, full history | 31 s | 3.5 GB |

Expect roughly 2.5x these on the full Norgate history. With 16 GB of RAM, start `grid.py` with 2–3 workers
for full-history runs.

## Tests

```
python -m pytest tests index_changes crash_rebound crash_short crash_bounce
```

Hand-built price paths cover every fill and exit rule, the eight template rules, no look-ahead, forward labels,
the Holdout lock and portfolio accounting; `tests/test_short.py` does the same for the short side.
