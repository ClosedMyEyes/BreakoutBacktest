"""
Shared configuration for the Trend Template backtest and scanner.
=================================================================
Every script imports paths and defaults from here so the backtester and the
live scanner can never drift apart (plan ground rule 3.5: one code path).

Edit DATA_DIR if the Parquet files should live somewhere other than ./store.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.environ.get("BB_DATA_DIR", os.path.join(ROOT_DIR, "store"))

PRICES_FILE   = os.path.join(DATA_DIR, "prices.parquet")     # Phase 1: one row per symbol-day
SYMBOLS_FILE  = os.path.join(DATA_DIR, "symbols.parquet")    # Phase 1: one row per symbol
MARKET_FILE   = os.path.join(DATA_DIR, "market.parquet")     # Phase 1: index series
FEATURES_FILE = os.path.join(DATA_DIR, "features.parquet")   # Phase 2: indicators + RS ranks
RANKS_FILE    = os.path.join(DATA_DIR, "ranks.parquet")      # Phase 2: cross-sectional ranks, row-aligned
LABELS_FILE   = os.path.join(DATA_DIR, "labels.parquet")     # Phase 2: forward returns (Mode A only), row-aligned
MARKET_FEATURES_FILE = os.path.join(DATA_DIR, "market_features.parquet")  # Phase 2/9: one row per day
VERSION_FILE  = os.path.join(DATA_DIR, "VERSION.json")       # last build stamp

# BB_WORK_DIR moves the split, research log and results elsewhere (used by tests)
WORK_DIR          = os.environ.get("BB_WORK_DIR", ROOT_DIR)
SPLIT_FILE        = os.path.join(WORK_DIR, "split.csv")         # Phase 3: Dev/Holdout, committed and locked
RESEARCH_LOG_FILE = os.path.join(WORK_DIR, "research_log.csv")  # ground rule 3.8
RESULTS_DIR       = os.path.join(WORK_DIR, "results")           # grid CSVs and trade logs

# =============================================================================
# NORGATE
# =============================================================================

NORGATE_DATABASES = ["US Equities", "US Equities Delisted"]

# Security subtypes kept as "common stock". Trial check (2026-10-08): subtype1
# is Equity / Exchange Traded Product / Derivative / Hybrid / Debt. Keep Equity,
# then drop subtype2 values listed below and names containing EXCLUDE_NAME_WORDS or "%".
# Run `python build_data.py --list-subtypes` to see subtype2 values and the count kept.
COMMON_STOCK_SUBTYPES = ["Equity"]
# Equity subtype2 on the trial: Operating/Holding Company 7144, Special Purpose
# Company 886 (SPACs: cash shells that don't trend), Investment Company 477
# (closed-end funds, BDCs). Keep operating companies only.
EXCLUDE_SUBTYPE2 = ["Special Purpose Company", "Investment Company"]
# Whole words only, so "UNITED" or "FUNDAMENTAL" don't match "UNIT" or "FUND"
EXCLUDE_NAME_WORDS = [
    "ETF", "ETN", "FUND", "PREFERRED", "PFD", "WARRANT", "WARRANTS", "WTS", "UNIT", "UNITS",
    "RIGHT", "RIGHTS", "NOTES", "DEBENTURE", "DEBENTURES",
]

INDEX_MEMBERSHIP = {            # column name -> Norgate index name
    "in_sp500": "S&P 500",
    "in_sp400": "S&P MidCap 400",
    "in_sp600": "S&P SmallCap 600",
    "in_r1000": "Russell 1000",
    "in_r2000": "Russell 2000",
}

MARKET_SYMBOLS = {              # column prefix -> Norgate symbol
    "spx":  "$SPX",
    "comp": "$COMP",            # Nasdaq Composite
    "spy":  "SPY",
    "qqq":  "QQQ",
}

CLASSIFICATION_SCHEME = "GICS"

# =============================================================================
# DATA RANGE
# =============================================================================

START_DATE = "1990-01-01"

# =============================================================================
# FEATURE DEFAULTS (Phase 2)
# =============================================================================

# Basic universe for RS ranking: every stock with a full RS lookback and an
# unadjusted close at or above this level that day. Ranking is market-wide, so
# it uses all stocks, Dev and Holdout alike (plan 4.4).
RS_RANK_MIN_PRICE = 1.0

BREAKOUT_LOOKBACKS = [20, 50, 252]
TIGHTNESS_WINDOWS  = [5, 10, 15]
FORWARD_HORIZONS   = [5, 20, 60]

# Price breaks (found on the Norgate trial, 2026-10-08). A day where Norgate's
# adjustment factor (unadjusted / adjusted close) moves, yet the adjusted close
# still jumps more than BREAK_MOVE_PCT, or any bar 10x up or 90% down. These are
# usually old shares swapped for new ones (a bankruptcy exit like WW in 2025, a
# de-SPAC) or a micro-cap reverse split. Indicators that look back across the
# break are meaningless, so rules.universe_mask blocks new entries for
# `block_after_break` bars afterwards.
BREAK_FACTOR_JUMP = 1.4     # factor moves by more than 1.4x either way
BREAK_MOVE_PCT    = 25.0
BREAK_EXTREME     = 10.0

# =============================================================================
# VALIDATION (Phase 3)
# =============================================================================

SPLIT_SEED = 20261007
WALK_FORWARD = {"train_years": 10, "test_years": 3, "step_years": 3}
MIN_TRADES_PER_VARIANT = 200

# =============================================================================
# COSTS (Phase 4)
# =============================================================================

DEFAULT_SLIPPAGE_PCT   = 0.20    # per side, % of price
COMMISSION_PER_SHARE   = 0.005   # USD, IBKR fixed-ish
