# -*- coding: utf-8 -*-
"""Export the per-year metrics of the screening experiments (already in each run's json) to csv.

Reads ~/ext/runs/p7s_<name>/seed_XX.json (LightGBM, p7_screen.py) and ~/ext/runs/p7m_<name>/seed_XX.json (MLP,
p7_screen_mlp.py) and writes results/external/p7_screen_years.csv and p7_screen_mlp_years.csv: one row per configuration,
seed and test year (2017-2020), with IC, Rank IC, annualized excess return after costs (in percent) and the number of days.
No model is run. Prefixes without runs are skipped.

Usage: ~/ext/.venv-qlib/bin/python external/p7_screen_years.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTD = HERE.parent / "results" / "external"
RUNS = Path.home() / "ext" / "runs"

for prefix, out in (("p7s_", "p7_screen_years.csv"), ("p7m_", "p7_screen_mlp_years.csv")):
    rows = []
    for d in sorted(RUNS.glob(prefix + "*")):
        for f in sorted(d.glob("seed_*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            for y, v in sorted(r["years"].items()):
                rows.append(dict(name=r["name"], seed=int(r["seed"]), year=int(y), IC=v["IC"], rank_IC=v["Rank IC"],
                                 ann=100 * v["ann_excess_w_cost"], n_days=v["n_days"]))
    if not rows:
        print(f"{prefix}: no runs found, skipped")
        continue
    with open(OUTD / out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len({(r["name"], r["seed"]) for r in rows})
    print(f"{prefix}: {n} runs, {len(rows)} rows -> {out}")
