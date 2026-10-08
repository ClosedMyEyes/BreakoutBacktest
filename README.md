# BreakoutBacktest

Backtest and (later) nightly scanner for the O'Neil / Minervini long-side Trend Template
system. It follows the build plan in `Trend_Template_System_Plan.docx` (project files):
phases, ground rules and decisions are referenced by number in the code.

This is a separate program from Stockbot and never runs inside the live trading process.

## Status

| Phase | What | State |
|---|---|---|
| 1 | `build_data.py`: Norgate → local Parquet, sanity checks | Written; tested on synthetic data |
| 2 | `build_features.py`: indicators, RS ranks, market features, forward labels | Written; tested |
| 3 | `make_split.py`: locked Dev/Holdout split; walk-forward windows in `grid.py` | Written; tested |
| 4 | `engine.py`, `setup_trend.py`, `grid.py`: fill model, exits, Modes B and C | Written; tested |
| 5 | `filter_test.py`: Mode A | Written; tested |
| 5.4 | Kill test: F1 + X2 vs date-matched F0 | `grid.py` is set up for it |
| 6 | Entry families | F0–F3 done; F4 (full VCP), F5 (flag), F6 (retest) to do |
| 7 | Exit models | X1, X2 (original or breakeven stop after the +20% hold), X3, X5, X6 and time exit done; X4 (partial + trail) to do |
| 8 | Mode C portfolio | Done, with Monte Carlo drawdown range |
| 9–11 | Market light, fundamentals, earnings dates | Market features computed; rules to do |
| 12–14 | Holdout run, scanner, forward test | To do |

Nothing has been run on real data yet. Every number produced so far comes from synthetic data and
means nothing about the strategy.

## Setup (Windows machine with Norgate)

```
pip install -r requirements.txt
```

The Norgate Data Updater must be running for `build_data.py`.

## Order of operations

```
python build_data.py --list-subtypes     # trial week 1: check which security subtypes are common stock,
                                         # then set COMMON_STOCK_SUBTYPES in config.py
python build_data.py --limit 200         # quick end-to-end check
python build_data.py                     # full pull (prints the Phase 1 sanity checks)
python show_data.py ABPO                 # look at one stock's raw bars and adjustment days
python show_data.py --find lehman        # search names; --delisted lists every delisted stock
python build_features.py                 # Phase 2
python check_features.py --random 5      # compare against TradingView (Phase 2 done-when)
python make_split.py                     # Phase 3, once; commit split.csv
python filter_test.py --breakdowns       # Mode A baseline
python setup_trend.py                    # Mode B: F1 50-day breakout + X2, vs F0
python setup_trend.py --mode portfolio   # Mode C
python grid.py                           # grid / walk-forward (edit the top of the file)
python research_log.py                   # what has been tried so far
```

Phase 4 dummy strategy (buy a 20-day high, exit after 20 days):

```
python setup_trend.py --template off --n 20 --exit time --time-bars 20 --stop-pct 0 --save-trades results/dummy.csv
python check_features.py --trades results/dummy.csv --n 10
```

Without Norgate, `python build_data.py --source synthetic` creates fake data so the whole pipeline can be run
and debugged. Set `BB_DATA_DIR` and `BB_WORK_DIR` to keep test data and logs out of the repo.

## Files

| File | Job |
|---|---|
| `config.py` | Paths, Norgate names, defaults (walk-forward lengths, min trades, costs) |
| `data_sources.py` | Norgate source and the synthetic source |
| `build_data.py` | Phase 1 |
| `build_features.py` | Phase 2. Writes `features`, `ranks`, `labels` (forward returns, Mode A only) and `market_features` |
| `make_split.py` | Phase 3 Dev/Holdout split |
| `datastore.py` | Loads only the columns and rows a run needs; holds the Holdout lock |
| `rules.py` | Universe filter, Trend Template, entry families. Shared with the future scanner |
| `engine.py` | Fill model, exits, F0 baseline, portfolio, metrics |
| `params.py` | Turns every rule default into a CLI flag so `grid.py` keys map onto them |
| `filter_test.py` | Mode A |
| `setup_trend.py` | Modes B and C |
| `grid.py` | Your grid runner, extended: windows, walk-forward report, holdout lock, research log |
| `research_log.py` | Research log (plan 3.8) |
| `show_data.py` | Raw-data lookups: one stock's bars and adjustment days, name search, delisted list |
| `check_features.py` | Hand checks against TradingView |

## Guard rails built in

- **No look-ahead.** Features on day t use bars up to t. Forward returns live in a separate file that only
  Mode A may load. A test changes all future bars and checks that no past feature moves.
- **Survivorship.** Delisted stocks stay in; trades in a stock that delists exit at its last price.
- **Holdout lock.** `holdout` and `all` universes refuse to load without `--unlock-holdout`.
- **Date-matched F0.** Every Mode B run reports `f0_avg_R` and `edge_R`: random template stocks on the
  same signal days, same risk, same exits.
- **Count the tries.** `grid.py` logs the number of combinations to the research log, and the walk-forward
  report shows train vs test so the honest (test) number is always next to the flattering one.
- **Mode A statistics by date** with t-stats over non-overlapping periods.

## Speed and memory (measured on synthetic data at about 40% of the real size)

| Step | Time | Peak RAM |
|---|---|---|
| Feature build (22.8M rows) | 3 min | — |
| Mode A, full history | 14 s | 2.2 GB |
| Mode B with 3 F0 seeds, full history | 14 s | 2.8 GB |
| Mode B, 10-year window | 8 s | 2.3 GB |
| Mode C, full history | 31 s | 3.5 GB |

Expect roughly 2.5x these on the full Norgate history. With 16 GB of RAM, start `grid.py` with 2–3 workers
for full-history runs.

## Tests

```
python -m pytest tests
```

Hand-built price paths cover every fill and exit rule, the eight template rules, no look-ahead, forward labels,
the Holdout lock and portfolio accounting.
