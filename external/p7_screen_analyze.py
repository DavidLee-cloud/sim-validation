# -*- coding: utf-8 -*-
"""Analysis of the P7 external screening experiment (external/p7_screen.py), following external/PREREG_P7_screen_zh.md.

Reads ~/ext/runs/p7s_<name>/seed_XX.json and results/external/p7_screen_adopted.json.
Writes results/external/p7_screen_seeds.csv (one row per configuration and seed) and results/external/p7_screen.md.

Screens: round-1 candidates against the base; round-2 candidates against the configuration adopted after round 1;
round-3 candidates against the one adopted after round 2. Paired differences are taken seed by seed.
  d_A = mean over seeds 0-2 (the seeds on which the baseline of rounds 2 and 3 was adopted)
  d_B = mean over seeds 12-14 (fresh seeds, never used for any adoption)
  e   = mean over seeds 3-11 (expansion seeds)
Quantities (for annualized excess return after costs, in pp, and for IC):
  P1  overstatement of each adopted configuration: its mean on seeds 0-2 minus its mean on seeds 3-11; the same gap for every
      configuration never adopted (control); Proposition 1 benchmark sigma_cfg / sqrt(3) * mu_7 (an upper bound with shared seeds).
  P2  rounds 2-3: mean of d_A - d_B over screens, with a t statistic over screens.
  P3  retention slope of e on the clean development estimate (round 1: d_A, the base was not selected; rounds 2-3: d_B),
      against the ex-ante prediction kappa_3 computed from development seeds only (sigma^2 = mean within-screen variance of the
      three development seeds; tau^2 = variance of the development estimates minus sigma^2 / 3).
  P4  rounds 2-3: slope of e on d_A against slope of e on d_B (same screens).
  P5  the P3 slope and the ex-ante kappa for IC against those for annualized excess return.
Bootstrap intervals resample screens (fixed seed 0).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTD = HERE.parent / "results" / "external"
RUNS = Path.home() / "ext" / "runs"
DEV, FRESH, EXP = [0, 1, 2], [12, 13, 14], list(range(3, 12))
MU7 = 1.3522  # expected maximum of 7 independent standard normals
rng = np.random.default_rng(0)


def load() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.glob("p7s_*")):
        for f in sorted(d.glob("seed_*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            rows.append(dict(name=r["name"], seed=int(r["seed"]), ann=100 * r["ann_excess_w_cost"], ic=r["IC"],
                             overrides=json.dumps(r["overrides"], sort_keys=True)))
    return pd.DataFrame(rows)


def screens(adopt: dict) -> list:
    from p7_screen import ROUNDS
    base = {1: "base", 2: adopt["1"]["name"], 3: adopt["2"]["name"]}
    return [(r, n, base[r]) for r in (1, 2, 3) for n in ROUNDS[r]]


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def boot(x, y, B=5000):
    x, y = np.asarray(x), np.asarray(y); n = len(x); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0:
            out.append(np.polyfit(x[i], y[i], 1)[0])
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main() -> None:
    df = load()
    adopt = json.loads((OUTD / "p7_screen_adopted.json").read_text(encoding="utf-8"))
    OUTD.mkdir(parents=True, exist_ok=True)
    df.sort_values(["name", "seed"]).to_csv(OUTD / "p7_screen_seeds.csv", index=False)
    W = {k: df.pivot(index="seed", columns="name", values=k) for k in ("ann", "ic")}
    L = ["# P7 external screening experiment (Qlib LightGBM, Alpha158, CSI 300)", "",
         f"Configurations: {df.name.nunique()}; runs: {len(df)}; seeds per configuration: {df.groupby('name').seed.nunique().min()}–{df.groupby('name').seed.nunique().max()}.", "",
         "Adopted: " + "; ".join(f"after round {r}: {adopt[r]['name']}" for r in sorted(adopt)), ""]
    # P1
    adopted_names = {adopt[r]["name"] for r in adopt}
    L += ["## P1 Overstatement on seeds 0-2 (annualized excess after costs, pp)", "",
          "| configuration | adopted after round | mean seeds 0-2 | mean seeds 3-11 | gap |", "|---|---|---|---|---|"]
    gaps = {}
    for n in W["ann"].columns:
        a, b = W["ann"].loc[DEV, n].mean(), W["ann"].loc[EXP, n].mean(); gaps[n] = a - b
        rr = ", ".join(r for r in sorted(adopt) if adopt[r]["name"] == n) or "—"
        L.append(f"| {n} | {rr} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
    ctrl = [g for n, g in gaps.items() if n not in adopted_names]
    sig_cfg = float(np.sqrt(W["ann"].var(ddof=1).mean()))
    L += ["", f"Never adopted ({len(ctrl)} configurations): mean gap {np.mean(ctrl):+.2f}, s.d. {np.std(ctrl, ddof=1):.2f}. "
          f"Adopted: " + ", ".join(f"{n} {gaps[n]:+.2f}" for n in sorted(adopted_names)) + ". "
          f"Proposition 1 benchmark (independent noise, m = 7): sigma_cfg / sqrt(3) * mu_7 = {sig_cfg:.2f} / 1.732 * {MU7} = {sig_cfg / math.sqrt(3) * MU7:.2f}.", ""]
    # screens
    rows = []
    for r, n, b in screens(adopt):
        if n not in W["ann"].columns or b not in W["ann"].columns:
            continue
        rec = dict(round=r, name=n, baseline=b)
        for k in ("ann", "ic"):
            dd = W[k][n] - W[k][b]
            rec[f"dA_{k}"] = dd.loc[DEV].mean(); rec[f"dB_{k}"] = dd.loc[FRESH].mean(); rec[f"e_{k}"] = dd.loc[EXP].mean()
            rec[f"vA_{k}"] = dd.loc[DEV].var(ddof=1); rec[f"vB_{k}"] = dd.loc[FRESH].var(ddof=1)
        rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(OUTD / "p7_screen_screens.csv", index=False)
    late = S[S["round"] > 1]
    # P2
    x = late.dA_ann - late.dB_ann
    L += ["## P2 Rounds 2-3: development estimate on the adoption seeds minus on fresh seeds (pp)", "",
          f"{len(late)} screens: mean {x.mean():+.2f}, s.d. {x.std(ddof=1):.2f}, t = {x.mean() / x.std(ddof=1) * math.sqrt(len(x)):+.2f}. "
          f"By round: " + "; ".join(f"round {r}: {g.dA_ann.sub(g.dB_ann).mean():+.2f} (baseline gap {gaps[adopt[str(r - 1)]['name']]:+.2f})" for r, g in late.groupby('round')), ""]
    # P3 / P5
    L += ["## P3, P5 Retention on clean development seeds, against the ex-ante prediction", "",
          "| metric | screens | observed slope (95% bootstrap) | ex-ante kappa_3 | sigma^2 (dev seeds) | tau^2 (dev seeds) |", "|---|---|---|---|---|---|"]
    for k, lab in (("ann", "annualized excess (pp)"), ("ic", "IC")):
        dc = np.where(S["round"] == 1, S[f"dA_{k}"], S[f"dB_{k}"]); vc = np.where(S["round"] == 1, S[f"vA_{k}"], S[f"vB_{k}"])
        s2 = float(np.mean(vc)); tau2 = max(float(np.var(dc, ddof=1)) - s2 / 3, 0.0); kap = tau2 / (tau2 + s2 / 3) if tau2 + s2 > 0 else float("nan")
        lo, hi = boot(dc, S[f"e_{k}"])
        L.append(f"| {lab} | {len(S)} | {slope(dc, S[f'e_{k}']):.2f} ({lo:.2f} to {hi:.2f}) | {kap:.2f} | {s2:.4g} | {tau2:.4g} |")
    # P4
    L += ["", "## P4 Rounds 2-3: retention slope on the adoption seeds against fresh seeds", "",
          "| metric | screens | slope of e on d_A | slope of e on d_B |", "|---|---|---|---|"]
    for k in ("ann", "ic"):
        L.append(f"| {k} | {len(late)} | {slope(late[f'dA_{k}'], late[f'e_{k}']):.2f} | {slope(late[f'dB_{k}'], late[f'e_{k}']):.2f} |")
    L += ["", "## Screens", "", "| " + " | ".join(S.columns) + " |", "|" + "---|" * len(S.columns)]
    L += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in S.itertuples(index=False)]
    (OUTD / "p7_screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
