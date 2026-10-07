"""
Research log (plan ground rule 3.8)
===================================
One row per run: date, question, phase, data split, parameters, how many
combinations were tried, key results, decision. grid.py appends automatically;
fill in the `decision` column by hand once you've looked at the results.

The `combinations` column is what keeps the multiple-testing count honest
(plan 3.9): sum it over a phase to know how many variants were tried.

    python research_log.py            # print the log and the running total of tries
"""

import csv
import json
import os

import pandas as pd

import config

COLUMNS = ["date", "question", "phase", "split", "script", "params", "combinations",
           "runs", "results", "results_file", "decision"]


def append(question, phase, split, script, params, combinations, runs, results,
           results_file="", decision=""):
    new = not os.path.exists(config.RESEARCH_LOG_FILE)
    with open(config.RESEARCH_LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({
            "date": pd.Timestamp.now().isoformat(timespec="seconds"),
            "question": question, "phase": phase, "split": split, "script": script,
            "params": json.dumps(params, default=str), "combinations": combinations,
            "runs": runs, "results": json.dumps(results, default=str),
            "results_file": results_file, "decision": decision,
        })


if __name__ == "__main__":
    if not os.path.exists(config.RESEARCH_LOG_FILE):
        print("No research log yet.")
    else:
        log = pd.read_csv(config.RESEARCH_LOG_FILE)
        pd.set_option("display.max_colwidth", 60)
        print(log[["date", "phase", "split", "question", "combinations", "decision"]].to_string())
        print("\nVariants tried per phase:")
        print(log.groupby("phase")["combinations"].sum().to_string())
