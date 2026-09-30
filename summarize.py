# -*- coding: utf-8 -*-
"""Aggregate a grid: mean over model seeds per cell, plus cross-seed Jaccard of the top-k sets.

    python summarize.py <grid> [--by key1,key2,...]

Needs only numpy (no torch).  Default grouping = all unit keys except model_seed.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = ["real_ic", "real_ic_sd", "true_ic", "oracle_capture", "chosen_epoch", "share_epoch_le2", "displacement",
        "displacement_from_first_init", "slope_real", "slope_true", "ann_net", "mdd", "sharpe", "excess_ann", "excess_ir",
        "bucket0_mu_ann", "bucket5_mu_ann", "bucket10_mu_ann", "bucket20_mu_ann",
        "bucket0_r_ann", "bucket5_r_ann", "bucket10_r_ann", "bucket20_r_ann"]


def jaccard(runs: list[dict]) -> float:
    vals = []
    for a, b in itertools.combinations(runs, 2):
        da = {d["day"]: set(d["top"]) for d in a["decisions"]}
        for d in b["decisions"]:
            if d["day"] in da:
                s, t = da[d["day"]], set(d["top"])
                vals.append(len(s & t) / len(s | t))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--by", default=None)
    ap.add_argument("--part", default="all", help="summary part: all | overlap")
    a = ap.parse_args()
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / a.grid).glob("*.json"))]
    runs = [r for r in runs if r["unit"]["kind"] == "train"]
    by = a.by.split(",") if a.by else [k for k in runs[0]["unit"] if k not in ("model_seed", "kind")]
    groups: dict[tuple, list] = defaultdict(list)
    for r in runs:
        groups[tuple(r["unit"].get(k) for k in by)].append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(by, key))
        row["n"] = len(rs)
        for k in KEYS:
            v = np.array([r["summary"][a.part].get(k, np.nan) for r in rs], dtype=float)
            row[k] = float(v.mean())
            if k == "ann_net":
                row["ann_net_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        row["jaccard"] = jaccard(rs)
        row["k_over_c"] = 1.0 / row["slope_real"] if row["slope_real"] > 0 else float("nan")
        rows.append(row)
    cols = by + ["n", "real_ic", "real_ic_sd", "true_ic", "oracle_capture", "chosen_epoch", "share_epoch_le2",
                 "displacement", "displacement_from_first_init", "slope_real", "k_over_c", "jaccard",
                 "ann_net", "ann_net_sd", "mdd", "excess_ann", "excess_ir", "bucket0_mu_ann", "bucket5_mu_ann", "bucket10_mu_ann",
                 "bucket0_r_ann", "bucket10_r_ann"]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in cols))
    out = ROOT / "results" / f"summary_{a.grid}_{a.part}.json"
    out.write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
