"""
Loading helpers shared by every test script and the live scanner.
================================================================
load_panel() reads only the requested columns from the row-aligned Parquet
files (prices, features, ranks, labels) and returns one DataFrame.

The Holdout lock lives here so no script can get around it: asking for the
holdout or all-stocks universe without unlock_holdout=True raises an error
(plan 4.4: nobody looks at Holdout results until Phase 12).
"""

import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config

ROW_FILES = [  # (path, is_label_file)
    (config.PRICES_FILE, False),
    (config.FEATURES_FILE, False),
    (config.RANKS_FILE, False),
    (config.LABELS_FILE, True),
]


class HoldoutLocked(Exception):
    pass


def check_universe(universe, unlock_holdout):
    if universe not in ("dev", "holdout", "all"):
        raise ValueError(f"universe must be dev, holdout or all, not {universe!r}")
    if universe in ("holdout", "all") and not unlock_holdout:
        raise HoldoutLocked(
            "The Holdout is locked until Phase 12. Pass --unlock-holdout only for the "
            "one final run, after the rules are frozen in the research log.")


def tradable_symbols(universe):
    """Symbols that may be traded in this universe. RS ranks and market
    features are always computed on every stock (plan 4.4)."""
    if universe == "all":
        return None
    if not os.path.exists(config.SPLIT_FILE):
        raise FileNotFoundError(f"{config.SPLIT_FILE} missing: run make_split.py first")
    split = pd.read_csv(config.SPLIT_FILE)
    return set(split.loc[split["split"] == universe, "symbol"])


def _schema_cols(path):
    return set(pq.read_schema(path).names) if os.path.exists(path) else set()


def load_panel(columns, start=None, end=None, universe="dev", unlock_holdout=False,
               tail_days=0, labels=False):
    """
    columns     feature/price/rank column names wanted (symbol and date always included)
    start, end  date range of interest (features are precomputed, so no warm-up
                rows are needed before start)
    tail_days   extra calendar days loaded after end so trades opened near the end
                of a window can still exit
    labels      allow forward-return columns (Mode A only)
    """
    check_universe(universe, unlock_holdout)
    columns = [c for c in dict.fromkeys(columns) if c not in ("symbol", "date")]

    # Row mask built once from date + symbol, applied to every row-aligned file
    # before converting to pandas, so Holdout rows never use memory
    key = pq.read_table(config.PRICES_FILE, columns=["symbol", "date"])
    mask = None
    if start is not None:
        mask = pc.greater_equal(key.column("date"), pd.Timestamp(start))
    if end is not None:
        hi = pd.Timestamp(end) + pd.Timedelta(days=tail_days)
        m2 = pc.less_equal(key.column("date"), hi)
        mask = m2 if mask is None else pc.and_(mask, m2)
    allowed = tradable_symbols(universe)
    if allowed is not None:
        sym = key.column("symbol")
        if pa.types.is_dictionary(sym.type):
            sym = sym.cast(pa.string())
        m3 = pc.is_in(sym, value_set=pa.array(sorted(allowed), type=pa.string()))
        mask = m3 if mask is None else pc.and_(mask, m3)
    del key

    parts, remaining = [], list(columns)
    for path, is_label in ROW_FILES:
        have = _schema_cols(path)
        want = [c for c in remaining if c in have]
        if path == config.PRICES_FILE:
            want = ["symbol", "date"] + want
        if not want:
            continue
        if is_label and not labels:
            raise ValueError(f"Columns {want} are forward returns; only Mode A may load them")
        parts.append(_read_filtered(path, want, mask).to_pandas())
        remaining = [c for c in remaining if c not in want]
    if remaining:
        raise KeyError(f"Unknown columns: {remaining}")

    df = pd.concat(parts, axis=1)
    # symbol stays categorical: far less memory than millions of strings
    df["symbol"] = df["symbol"].astype("category").cat.remove_unused_categories()
    return df


def _read_filtered(path, columns, mask, batch_rows=2_000_000):
    """Read in batches and drop masked rows as we go, so peak memory is the
    filtered result plus one batch rather than the whole file."""
    if mask is None:
        return pq.read_table(path, columns=columns)
    mask = mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask
    out, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        m = mask.slice(offset, batch.num_rows)
        offset += batch.num_rows
        out.append(batch.filter(m))
    return pa.Table.from_batches(out)


def load_market(columns=None):
    path = config.MARKET_FEATURES_FILE
    return pq.read_table(path, columns=(["date"] + columns) if columns else None).to_pandas()


def load_symbols():
    return pd.read_parquet(config.SYMBOLS_FILE)
