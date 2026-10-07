"""
Command-line flags generated from the rule defaults
===================================================
Every key in the *_DEFAULTS dicts becomes a --flag (underscores -> dashes), so
grid.py PARAM_GRID keys map straight onto them, exactly like the setup scripts
you already use.
"""

import argparse

from engine import EXIT_DEFAULTS, PORTFOLIO_DEFAULTS
from rules import ENTRY_DEFAULTS, TEMPLATE_DEFAULTS, UNIVERSE_DEFAULTS

GROUPS = {
    "universe":  UNIVERSE_DEFAULTS,
    "template":  TEMPLATE_DEFAULTS,
    "entry":     ENTRY_DEFAULTS,
    "exit":      EXIT_DEFAULTS,
    "portfolio": PORTFOLIO_DEFAULTS,
}


def add_param_flags(parser, groups):
    seen = set()
    for g in groups:
        for key, default in GROUPS[g].items():
            if key in seen:
                continue
            seen.add(key)
            flag = "--" + key.replace("_", "-")
            if isinstance(default, bool):
                parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                                    default=default)
            elif isinstance(default, int):
                parser.add_argument(flag, type=float, default=default)   # grids may pass 63.0
            elif isinstance(default, float):
                parser.add_argument(flag, type=float, default=default)
            else:
                parser.add_argument(flag, type=str, default=default)


def add_run_flags(parser):
    parser.add_argument("--start", type=str, default=None, help="First signal date")
    parser.add_argument("--end", type=str, default=None, help="Last signal date")
    parser.add_argument("--universe", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="Required for holdout/all. Phase 12 only.")
    parser.add_argument("--output-json", action="store_true",
                        help="Print GRID_RESULT:<json> at the end (used by grid.py)")


def collect(args, groups):
    out = {}
    for g in groups:
        for key, default in GROUPS[g].items():
            val = getattr(args, key)
            if isinstance(default, int) and not isinstance(default, bool):
                val = int(round(val))
            out[key] = val
    return out
