# -*- coding: utf-8 -*-
"""Module A verdicts exactly as pre-registered (PREREG_A_zh.md, section 3).  Needs numpy + scipy only.

    python analyze_A.py            # prints every statistic and a PASS/FAIL line per hypothesis
Writes results/analysis_A.json.  Grids that are not complete are reported as missing, never partially judged.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
EMP = dict(ic=0.15, drift="regime")
DRIFTS = ["none", "rw_slow", "rw_fast", "regime"]


def load(grid):
    d = ROOT / "results" / grid
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.exists() else []


def val(r, k, part="all"):
    return float(r["summary"][part][k])


def jaccard(runs):
    out = []
    for a, b in itertools.combinations(runs, 2):
        da = {x["day"]: set(x["top"]) for x in a["decisions"]}
        for x in b["decisions"]:
            if x["day"] in da:
                s, t = da[x["day"]], set(x["top"])
                out.append(len(s & t) / len(s | t))
    return float(np.mean(out))


def holm(ps):
    order = np.argsort(ps)
    adj, m, run = np.empty(len(ps)), len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    out, pvals = {}, {}
    A = load("A_main")
    idx = {(r["unit"]["ic"], r["unit"]["drift"], r["unit"]["protocol"], r["unit"]["data_seed"],
            r["unit"]["model_seed"], r["unit"]["start_day"]): r for r in A}
    ics = sorted({k[0] for k in idx})
    if len(A) == 2240:
        # A1 ① share of <=2 epochs for P1 decreasing in IC (per drift)
        rho = {}
        for dr in DRIFTS:
            share = [np.mean([val(idx[(ic, dr, "p1", ds, ms, 1000)], "share_epoch_le2")
                              for ds in (0, 1) for ms in range(5)]) for ic in ics]
            rho[dr] = (share, float(stats.spearmanr(ics, share).correlation))
        # A1 ② paired diff at the empirical cell
        pairs = [val(idx[(EMP["ic"], EMP["drift"], "p3", ds, ms, st)], "true_ic")
                 - val(idx[(EMP["ic"], EMP["drift"], "p1", ds, ms, st)], "true_ic")
                 for ds in (0, 1) for ms in range(5) for st in (1000, 2000)]
        t = stats.ttest_1samp(pairs, 0.0, alternative="greater")
        pvals["A1"] = float(t.pvalue)
        out["A1"] = {"p1_share_le2_by_ic": {k: v[0] for k, v in rho.items()},
                     "spearman_by_drift": {k: v[1] for k, v in rho.items()},
                     "emp_cell_diff_mean": float(np.mean(pairs)), "t": float(t.statistic), "p": float(t.pvalue),
                     "pass_part1": all(v[1] < 0 for v in rho.values())}
        # A1b and A2: P3-P1 true IC diff by (ic, drift), start 1000
        diff = {(ic, dr): float(np.mean([val(idx[(ic, dr, "p3", ds, ms, 1000)], "true_ic")
                                         - val(idx[(ic, dr, "p1", ds, ms, 1000)], "true_ic")
                                         for ds in (0, 1) for ms in range(5)])) for ic in ics for dr in DRIFTS}
        by_ic = [float(np.mean([diff[(ic, dr)] for dr in DRIFTS])) for ic in ics]
        out["A1b"] = {"diff_by_ic": dict(zip(map(str, ics), by_ic)), "pass": by_ic[-1] < by_ic[0]}
        a2 = {}
        for ic in (0.10, 0.15):
            d = [diff[(ic, dr)] for dr in ("none", "rw_slow", "rw_fast")]
            a2[str(ic)] = {"none/slow/fast": d, "full_order": d[0] < d[1] < d[2], "none_lt_fast": d[0] < d[2]}
        out["A2"] = {**a2, "pass": any(v["full_order"] for v in a2.values()) and all(
            v["full_order"] or v["none_lt_fast"] for v in a2.values())}
        # A3 start dependence (regime cells, all ics; test at the empirical cell) and displacement
        a3, emp_p = {}, None
        for ic in ics:
            dp = {pr: [abs(val(idx[(ic, "regime", pr, ds, ms, 1000)], "excess_ann", "overlap")
                           - val(idx[(ic, "regime", pr, ds, ms, 2000)], "excess_ann", "overlap"))
                       for ds in (0, 1) for ms in range(5)] for pr in ("p1", "p3")}
            w = stats.wilcoxon(dp["p1"], dp["p3"], alternative="greater")
            a3[str(ic)] = {"p1_mean_absdiff": float(np.mean(dp["p1"])), "p3_mean_absdiff": float(np.mean(dp["p3"])),
                           "p": float(w.pvalue)}
            if ic == EMP["ic"]:
                emp_p = float(w.pvalue)
        pvals["A3"] = emp_p
        disp_ok = all(val(idx[(ic, dr, "p1", ds, ms, 1000)], "displacement")
                      < val(idx[(ic, dr, "p3", ds, ms, 1000)], "displacement")
                      for ic in ics for dr in DRIFTS for ds in (0, 1) for ms in range(5))
        out["A3"] = {"by_ic": a3, "displacement_p1_lt_p3_all": disp_ok}
        # A4 (part: Jaccard increasing in IC for P3)
        jac = {dr: [jaccard([idx[(ic, dr, "p3", ds, ms, 1000)] for ms in range(5)]) for ic in ics] for dr in DRIFTS}
        mono = {dr: all(b >= a for a, b in zip(v, v[1:])) for dr, v in jac.items()}
        out["A4_ic"] = {"jaccard_p3_by_ic": jac, "monotone": mono, "pass": sum(mono.values()) >= 3}
    else:
        out["A_main"] = f"incomplete ({len(A)}/2240)"

    H = load("A_hl")
    if len(H) == 480:
        hidx = defaultdict(list)
        for r in H:
            u = r["unit"]
            hidx[(u["drift"], u["style_share"], u["protocol"])].append(r)
        hls = ["p4_252", "p4_504", "p4_1008", "p3"]
        cells, mono_n, best = {}, 0, {}
        for dr in DRIFTS:
            for ss in (0.0, 0.3, 0.6):
                j = [float(np.mean([jaccard([r for r in hidx[(dr, ss, pr)] if r["unit"]["data_seed"] == ds])
                                    for ds in (0, 1)])) for pr in hls]
                ir = [float(np.mean([val(r, "excess_ir") for r in hidx[(dr, ss, pr)]])) for pr in hls]
                m = all(b >= a for a, b in zip(j, j[1:]))
                mono_n += m
                cells[f"{dr}/{ss}"] = {"jaccard": j, "excess_ir": ir, "monotone": m}
                if ss == 0.3:
                    best[dr] = [252, 504, 1008, float("inf")][int(np.argmax(ir))]
        out["A4_hl"] = {"cells": cells, "monotone_cells": mono_n, "pass": mono_n >= 8}
        out["A5"] = {"best_halflife_style0.3": best,
                     "pass": best["none"] >= best["rw_slow"] >= best["rw_fast"]}
    else:
        out["A_hl"] = f"incomplete ({len(H)}/480)"

    V = load("A_val")
    if len(V) == 120:
        g = defaultdict(list)
        for r in V:
            u = r["unit"]
            g[(u["ic"], u["drift"], u["val_days"])].append(val(r, "share_epoch_le2"))
        res, n_ok = {}, 0
        for ic in (0.10, 0.15):
            for dr in ("rw_slow", "regime"):
                s = [float(np.mean(g[(ic, dr, v)])) for v in (60, 120, 250)]
                ok = s[0] > s[1] > s[2]
                n_ok += ok
                res[f"{ic}/{dr}"] = {"share_le2_60_120_250": s, "decreasing": ok}
        out["A6"] = {"cells": res, "pass": n_ok >= 3}
    else:
        out["A_val"] = f"incomplete ({len(V)}/120)"

    E = load("A_e2e")
    if len(E) == 160:
        eidx = {(r["unit"]["ic"], r["unit"]["drift"], r["unit"]["protocol"], r["unit"]["data_seed"],
                 r["unit"]["model_seed"], r["unit"]["start_day"]): r for r in E}
        res, n_ok = {}, 0
        for ic in (0.15, 0.20):
            for dr in ("rw_slow", "regime"):
                d_ir = np.mean([val(eidx[(ic, dr, "p3", ds, ms, 1000)], "excess_ir")
                                - val(eidx[(ic, dr, "p1", ds, ms, 1000)], "excess_ir") for ds in (0, 1) for ms in range(5)])
                d_ic = np.mean([val(eidx[(ic, dr, "p3", ds, ms, 1000)], "true_ic")
                                - val(eidx[(ic, dr, "p1", ds, ms, 1000)], "true_ic") for ds in (0, 1) for ms in range(5)])
                ok = d_ir > 0 and d_ic > 0
                n_ok += ok
                res[f"{ic}/{dr}"] = {"diff_excess_ir": float(d_ir), "diff_true_ic": float(d_ic), "both_positive": ok}
        pairs = [val(eidx[(0.15, "regime", "p3", ds, ms, st)], "excess_ir")
                 - val(eidx[(0.15, "regime", "p1", ds, ms, st)], "excess_ir")
                 for ds in (0, 1) for ms in range(5) for st in (1000, 2000)]
        t = stats.ttest_1samp(pairs, 0.0, alternative="greater")
        pvals["A7"] = float(t.pvalue)
        out["A7"] = {"cells": res, "cells_pass": n_ok >= 3, "emp_t": float(t.statistic), "emp_p": float(t.pvalue)}
    else:
        out["A_e2e"] = f"incomplete ({len(E)}/160)"

    if len(pvals) == 3:
        keys = sorted(pvals)
        adj = holm(np.array([pvals[k] for k in keys]))
        out["holm"] = {k: {"p": pvals[k], "p_holm": float(a), "pass": a < 0.05} for k, a in zip(keys, adj)}
    (ROOT / "results" / "analysis_A.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
