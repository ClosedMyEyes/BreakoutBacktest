"""
Grid Search Runner — Trend Template system
==========================================
Your grid.py, extended for the market-wide backtest (plan 4.3). Same idea:
every combination of PARAM_GRID runs in parallel as a subprocess, each run
prints GRID_RESULT:<json>, results are sorted in the terminal and saved to CSV.

HOW TO USE:
  1. Pick BACKTESTER_FILE: filter_test.py (Mode A) or setup_trend.py (Modes B, C)
  2. Edit PARAM_GRID and FIXED_ARGS below
  3. Run: python grid.py

WHAT'S NEW vs the intraday grid.py:
  - WINDOWS: run every combination on several date windows. "walkforward"
    builds train/test windows from config.WALK_FORWARD and reports, for each
    fold, the best combination on the TRAIN window and how it did on the TEST
    window. Only those test-window numbers estimate the edge (plan 3.9).
  - Universe: FIXED_ARGS["UNIVERSE"] is dev, holdout or all. Holdout and all
    refuse to run unless UNLOCK_HOLDOUT = True (Phase 12 only).
  - SORT_METRIC: avg_R / edge_R for signal mode, mar for portfolio mode,
    edge20 for Mode A.
  - Trade logs for the top SAVE_TOP_TRADES combinations are saved.
  - Every grid run appends a row to research_log.csv (plan 3.8), including the
    number of combinations tried.

PARAM_GRID CONVENTIONS (unchanged):
  - Keys are the CLI flag names with dashes replaced by underscores, upper case.
  - Only uncomment what you want to vary; each extra list multiplies the runs.

WORKERS / MEMORY:
  Each worker loads the columns it needs for the whole market. Start with 3-4
  workers on the full Norgate data and watch RAM; reduce if it gets tight.
"""

import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import time

import pandas as pd

import config
import research_log

# =============================================================================
# CONFIGURATION — edit here
# =============================================================================

BACKTESTER_FILE = "setup_trend.py"   # or "filter_test.py" for Mode A
NUM_WORKERS     = 4
OUTPUT_CSV      = os.path.join(config.RESULTS_DIR, "grid_results.csv")
QUESTION        = "Kill test: F1 50-day breakout + X2 vs date-matched F0"   # goes in the research log
PHASE           = "5.4"

FIXED_ARGS = {                        # passed to every run
    "UNIVERSE":    "dev",
    "MODE":        "signal",          # signal (Mode B) / portfolio (Mode C); setup_trend.py only
    "F0_SEEDS":    3,
}
UNLOCK_HOLDOUT  = False               # Phase 12 only

WINDOWS = None                        # None = full history; "walkforward"; or a list of
                                      # ("YYYY-MM-DD", "YYYY-MM-DD") pairs
SORT_METRIC     = "avg_R"
MIN_TRADES      = config.MIN_TRADES_PER_VARIANT
SAVE_TOP_TRADES = 3

PARAM_GRID = {

    # ── Universe (Mode A knobs, plan 5.2) ────────────────────────────────────
    # "MIN_PRICE":       [5, 10, 15],
    # "LIQ_TOP_PCT":     [50, 30, 20],
    # "SIZE":            ["all", "sp500", "sp400", "sp600", "sp400_600"],

    # ── Trend Template ───────────────────────────────────────────────────────
    # "TREND_DAYS":      [21, 63, 105],
    # "HILO":            ["close", "intraday"],
    # "MIN_ABOVE_LOW":   [25, 30, 50, 100],
    # "MAX_BELOW_HIGH":  [15, 25, 35],
    # "MIN_RS":          [60, 70, 80, 90],
    # "RS_COL":          ["rs_rank", "rs_rank_252", "rs_rank_126"],

    # ── Entry family (plan Phase 6) ──────────────────────────────────────────
    "FAMILY":          ["f1"],
    "N":               [50],
    # "K": [5, 10, 15], "T": [5, 8, 12], "X": [10, 15],          # F2
    # "R": [0.5, 0.65, 0.8], "V": [0.6, 0.8],                    # F3
    # "VOL_MULT": [1.0, 1.4, 1.8], "VOL_TIMING": ["b", "c"],
    # "MAX_CHASE":       [5],

    # ── Exit (plan Phase 7) ──────────────────────────────────────────────────
    "EXIT":            ["x2"],
    "HOLD_STOP":       ["initial", "breakeven"],  # x2: stop after the +20% hold starts
    "STOP_PCT":        [7.5],
    "TARGET_PCT":      [22.5],
    "SLIPPAGE_PCT":    [0.20],
    # "TRAIL":           ["sma50", "ema21", "sma10"],              # x3
    # "STOP_MODE":       ["pct", "structural"],

    # ── Portfolio (Mode C, plan Phase 8) ─────────────────────────────────────
    # "MAX_POSITIONS":   [5, 8, 12],
    # "RISK_PCT":        [0.5, 0.75, 1.0],
    # "RANK_BY":         ["rs", "tight", "random"],
}

# =============================================================================
# RUNNER — no need to edit below this line
# =============================================================================

def to_flag(key):
    return "--" + key.lower().replace("_", "-")


def combo_to_cli_args(param_combo):
    """Convert dict like {"MIN_RS": 80} → ["--min-rs", "80"]."""
    cli = []
    for key, val in param_combo.items():
        cli += [to_flag(key), str(val)]
    return cli


def base_cli():
    cli = []
    for key, val in FIXED_ARGS.items():
        if key in ("MODE", "F0_SEEDS") and BACKTESTER_FILE != "setup_trend.py":
            continue
        cli += [to_flag(key), str(val)]
    if UNLOCK_HOLDOUT:
        cli.append("--unlock-holdout")
    return cli


def run_combination(args):
    """Worker: runs the backtester once, returns its stats dict."""
    param_combo, window, backtester_path, extra = args
    cmd = [sys.executable, backtester_path, "--output-json"] + base_cli() + combo_to_cli_args(param_combo)
    if window:
        cmd += ["--start", window[0], "--end", window[1]]
    cmd += extra
    tag = {"window": f"{window[0]}..{window[1]}" if window else "all"}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        for line in result.stdout.splitlines():
            if line.startswith("GRID_RESULT:"):
                stats = json.loads(line[len("GRID_RESULT:"):])
                print(".", end="", flush=True)
                return {**param_combo, **tag, **stats, "error": None}
        err = result.stderr.strip().splitlines()
        print("x", end="", flush=True)
        return {**param_combo, **tag, "error": err[-1] if err else "no output"}
    except subprocess.TimeoutExpired:
        print("T", end="", flush=True)
        return {**param_combo, **tag, "error": "timeout"}
    except Exception as e:
        print("x", end="", flush=True)
        return {**param_combo, **tag, "error": str(e)}


def walk_forward_windows():
    """Train/test folds from config.WALK_FORWARD over the data's date range."""
    dates = pd.read_parquet(config.MARKET_FILE, columns=["date"])["date"]
    first, last = dates.min().year, dates.max().year
    wf = config.WALK_FORWARD
    folds, y = [], first
    while y + wf["train_years"] <= last:
        train = (f"{y}-01-01", f"{y + wf['train_years'] - 1}-12-31")
        ty = y + wf["train_years"]
        test = (f"{ty}-01-01", f"{min(ty + wf['test_years'] - 1, last)}-12-31")
        folds.append((train, test))
        y += wf["step_years"]
    return folds


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backtester_path = os.path.join(script_dir, BACKTESTER_FILE)
    if not os.path.exists(backtester_path):
        print(f"ERROR: Could not find '{BACKTESTER_FILE}' in {script_dir}")
        sys.exit(1)
    if FIXED_ARGS.get("UNIVERSE", "dev") != "dev" and not UNLOCK_HOLDOUT:
        print("ERROR: the Holdout is locked until Phase 12. Set UNIVERSE to dev, "
              "or UNLOCK_HOLDOUT = True for the one final run.")
        sys.exit(1)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    folds = None
    if WINDOWS == "walkforward":
        folds = walk_forward_windows()
        windows = sorted({w for f in folds for w in f})
    elif WINDOWS:
        windows = list(WINDOWS)
    else:
        windows = [None]

    jobs = [(c, w, backtester_path, []) for c in combos for w in windows]
    total = len(jobs)

    print("=" * 60)
    print(f"GRID SEARCH  —  {len(combos)} combinations x {len(windows)} window(s) = {total} runs"
          f"  —  {NUM_WORKERS} workers")
    print(f"Backtester: {BACKTESTER_FILE}   Fixed: {FIXED_ARGS}")
    print("Parameters:")
    for k, v in PARAM_GRID.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("Running... (dots = completed runs,  x = error,  T = timeout)\n")

    start = time.time()
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(run_combination, jobs)
    elapsed = time.time() - start
    print()

    df = pd.DataFrame(results)
    errors = df[df["error"].notna()]
    good = df[df["error"].isna()].copy()

    count_col = "trades" if "trades" in good else ("signals" if "signals" in good else None)
    if len(good) and SORT_METRIC in good:
        good[SORT_METRIC] = pd.to_numeric(good[SORT_METRIC], errors="coerce")
        if "total_R" in good and "max_dd" in good:
            good["R_per_DD"] = (pd.to_numeric(good["total_R"]) /
                                pd.to_numeric(good["max_dd"]).abs()).round(3)
        good["enough_trades"] = good[count_col] >= MIN_TRADES if count_col else True
        good = good.sort_values(["enough_trades", SORT_METRIC], ascending=[False, False]).reset_index(drop=True)

    df = pd.concat([good, errors], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", "{:.3f}".format)
    stat_pref = ["window", "trades", "signals", "taken", "win_rate", "avg_R", "median_R", "f0_avg_R",
                 "edge_R", "payoff", "total_R", "max_dd", "R_per_DD", "top10_share", "avg_hold",
                 "cagr", "max_dd_pct", "mar", "sharpe",
                 "pass_rate", "edge5", "edge20", "t20", "edge60", "t60", "big_winner_pass"]
    show = keys + [c for c in stat_pref if c in df.columns]
    print("\n" + "=" * 60)
    print(f"RESULTS  (sorted by {SORT_METRIC}; runs under {MIN_TRADES} trades sorted last)")
    print("=" * 60)
    print(df[show].head(60).to_string(index=True))
    print(f"\nCompleted {total} runs in {elapsed:.1f}s  ({elapsed / max(total, 1):.1f}s avg per run)")
    if len(errors):
        print(f"\n{len(errors)} run(s) errored or timed out. First error: {errors['error'].iloc[0]}")
    print(f"\nFull results saved to: {OUTPUT_CSV}")

    summary = {}
    if folds and len(good):
        summary = walk_forward_report(good, keys, folds)
    elif len(good):
        best = good.iloc[0]
        print(f"\nBEST COMBINATION (by {SORT_METRIC}):")
        for k in keys:
            print(f"  {k} = {best[k]}")
        for m in ["trades", "avg_R", "edge_R", "f0_avg_R", "total_R", "max_dd", "mar", "edge20", "t20"]:
            if m in best and pd.notna(best[m]):
                print(f"  → {m:<9}: {best[m]}")
        summary = {k: best[k] for k in keys} | {SORT_METRIC: best[SORT_METRIC]}
        if SAVE_TOP_TRADES and BACKTESTER_FILE == "setup_trend.py" and not folds:
            save_top_trades(good, keys, backtester_path)

    research_log.append(
        question=QUESTION, phase=PHASE, split=FIXED_ARGS.get("UNIVERSE", "dev"),
        script=BACKTESTER_FILE, params={"grid": PARAM_GRID, "fixed": FIXED_ARGS, "windows": WINDOWS},
        combinations=len(combos), runs=total, results=summary, results_file=OUTPUT_CSV,
    )
    print(f"Research log updated: {config.RESEARCH_LOG_FILE}")


def walk_forward_report(good, keys, folds):
    """For each fold: best combination on train -> its result on test."""
    rows = []
    count_col = "trades" if "trades" in good else ("signals" if "signals" in good else None)
    for train, test in folds:
        tr = good[good["window"] == f"{train[0]}..{train[1]}"]
        if count_col:
            tr = tr[tr[count_col] >= MIN_TRADES]
        if tr.empty:
            continue
        best = tr.sort_values(SORT_METRIC, ascending=False).iloc[0]
        te = good[(good["window"] == f"{test[0]}..{test[1]}")]
        for k in keys:
            te = te[te[k] == best[k]]
        if te.empty:
            continue
        rows.append({"train": f"{train[0][:4]}-{train[1][:4]}", "test": f"{test[0][:4]}-{test[1][:4]}",
                     **{k: best[k] for k in keys},
                     f"train_{SORT_METRIC}": best[SORT_METRIC],
                     f"test_{SORT_METRIC}": te.iloc[0][SORT_METRIC],
                     "test_trades": te.iloc[0].get(count_col) if count_col else None})
    if not rows:
        print("\nWalk-forward: no fold had a combination with enough trades.")
        return {}
    wf = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("WALK-FORWARD  (best on train → result on the following test window)")
    print("=" * 60)
    print(wf.to_string(index=False))
    test_mean = pd.to_numeric(wf[f"test_{SORT_METRIC}"]).mean()
    train_mean = pd.to_numeric(wf[f"train_{SORT_METRIC}"]).mean()
    print(f"\nAverage {SORT_METRIC}: train {train_mean:.3f}  →  test {test_mean:.3f}")
    print("The TEST number is the honest estimate. A big drop from train means overfitting.")
    return {f"wf_test_{SORT_METRIC}": round(float(test_mean), 4),
            f"wf_train_{SORT_METRIC}": round(float(train_mean), 4), "folds": len(wf)}


def save_top_trades(good, keys, backtester_path):
    eligible = good[good.get("enough_trades", True)] if "enough_trades" in good else good
    for rank, (_, row) in enumerate(eligible.head(SAVE_TOP_TRADES).iterrows(), 1):
        path = os.path.join(config.RESULTS_DIR, f"trades_top{rank}.csv")
        combo = {k: row[k] for k in keys}
        run_combination((combo, None, backtester_path, ["--save-trades", path]))
        print(f"\nTrade log for #{rank} saved to {path}")


if __name__ == "__main__":
    main()
