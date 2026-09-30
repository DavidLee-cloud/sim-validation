# -*- coding: utf-8 -*-
"""Run one chunk of a simulation grid; every unit writes results/<grid>/<index>.json and is skipped if present.

    python run_grid.py <grid> [--chunk C --n-chunks K] [--limit N] [--list]

Units are enumerated in a fixed order, so chunk C takes indices i with i % K == C.  Panels are cached under
``SIM_CACHE`` (default ~/.sim_cache) because many units share one simulated panel.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sim.dgp import DGPConfig, simulate  # noqa: E402
from sim.models import LossSpec  # noqa: E402
from sim.protocols import RunSpec, backtest, summarize  # noqa: E402
from sim.score_level import rank_curve  # noqa: E402

CACHE = Path(os.environ.get("SIM_CACHE", Path.home() / ".sim_cache"))
INF = float("inf")


def _prod(**axes):
    keys = list(axes)
    for vals in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, vals))


# --------------------------------------------------------------------------- grids
def grid_units(name: str) -> list[dict]:
    if name == "calib":        # fingerprint calibration around the empirical cell (design 2.3)
        # ic is the oracle IC (truth vs realised); the empirical fingerprints refer to the model's realised IC
        return list(_prod(kind=["train"], ic=[0.05, 0.10, 0.15, 0.20], style_vol_h=[0.02, 0.04],
                          drift=["regime"], style_share=[0.3], tail_df=[6.0], protocol=["p1", "p3"],
                          model=["reg"], data_seed=[0], model_seed=[0, 1, 2], start_day=[1000]))
    if name == "calib2":       # after calib: style_vol_h 0.04 matches the IC volatility; P1 now rolling 252/120
        return list(_prod(kind=["train"], ic=[0.10, 0.15, 0.20], style_vol_h=[0.04], drift=["regime"],
                          style_share=[0.3], tail_df=[6.0], protocol=["p1", "p1x", "p3"], model=["reg"],
                          data_seed=[0], model_seed=[0, 1, 2], start_day=[1000]))
    if name == "calib_e2e":    # k/c inflation and top-of-ranking curve of decision-focused training
        return list(_prod(kind=["train"], ic=[0.15, 0.20], style_vol_h=[0.04], drift=["regime"], style_share=[0.3],
                          tail_df=[6.0], lowvol_share=[0.2], protocol=["p1", "p3"], model=["e2e", "reg"],
                          data_seed=[0], model_seed=[0, 1, 2], start_day=[1000]))
    if name == "A_main":       # A1-A3
        return list(_prod(kind=["train"], ic=[0.02, 0.05, 0.10, 0.20], drift=["none", "rw_slow", "rw_fast", "regime"],
                          style_share=[0.3], tail_df=[6.0], protocol=["p1", "p1c", "p2", "p3", "p4_504", "p5"],
                          model=["reg"], data_seed=[0, 1], model_seed=[0, 1, 2, 3, 4], start_day=[1000, 2000]))
    if name == "A_val":        # validation length for P1
        return list(_prod(kind=["train"], ic=[0.05, 0.10], drift=["rw_slow", "regime"], style_share=[0.3],
                          tail_df=[6.0], protocol=["p1"], val_days=[60, 120, 250], model=["reg"],
                          data_seed=[0, 1], model_seed=[0, 1, 2, 3, 4], start_day=[1000]))
    if name == "A_hl":         # A4-A5: effective sample, style regimes, seed divergence
        return list(_prod(kind=["train"], ic=[0.05], drift=["none", "rw_slow", "rw_fast", "regime"],
                          style_share=[0.0, 0.3, 0.6], tail_df=[6.0],
                          protocol=["p4_252", "p4_504", "p4_1008", "p3"], model=["reg"], data_seed=[0, 1],
                          model_seed=[0, 1, 2, 3, 4], start_day=[1000]))
    if name == "A_e2e":        # transfer to decision-focused training
        return list(_prod(kind=["train"], ic=[0.05, 0.10], drift=["rw_slow", "regime"], style_share=[0.3],
                          tail_df=[6.0], protocol=["p1", "p3"], model=["e2e"], data_seed=[0, 1],
                          model_seed=[0, 1, 2, 3, 4], start_day=[1000, 2000]))
    if name == "B_train":      # B4-B5: method or market; fidelity
        return list(_prod(kind=["train"], ic=[0.05, 0.10], drift=["none", "regime"], style_share=[0.3],
                          tail_df=[3.0, 6.0], lowvol_share=[0.2], protocol=["p3"],
                          model=["reg", "e2e", "e2e_fid"], data_seed=[0, 1], model_seed=[0, 1, 2, 3, 4],
                          start_day=[1000]))
    if name == "B_score":      # B1-B3
        u = list(_prod(kind=["score"], case=["b1"], noise=[1.0, 3.0, 10.0], b=[0.0], df=[INF]))
        u += list(_prod(kind=["score"], case=["b2"], noise=[3.0], b=[0.0, 0.5, 1.0, 2.0], df=[3.0, 6.0, INF]))
        u += list(_prod(kind=["score"], case=["b3"], noise=[1.0, 3.0], b=[0.0], df=[INF]))
        return u
    raise SystemExit(f"unknown grid {name}")


# --------------------------------------------------------------------------- execution
def _panel(u: dict):
    fields = set(DGPConfig.__dataclass_fields__)
    kw = {k: v for k, v in u.items() if k in fields and k != "seed"}
    cfg = DGPConfig(**kw, seed=int(u["data_seed"]))
    key = hashlib.sha1(json.dumps(cfg.as_dict(), sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"panel_{key}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    CACHE.mkdir(parents=True, exist_ok=True)
    panel = simulate(cfg)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(panel, fh, protocol=4)
    tmp.replace(path)
    return panel


def _env() -> dict:
    import platform
    import numpy
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "platform": platform.platform()}


def run_unit(u: dict) -> dict:
    t0 = time.perf_counter()
    if u["kind"] == "score":
        out = rank_curve(u["case"], noise=u["noise"], b=u["b"], df=u["df"])
        return {"unit": u, "summary": out, "sec": round(time.perf_counter() - t0, 1)}
    panel = _panel(u)
    model = "e2e" if u["model"].startswith("e2e") else "reg"
    loss = LossSpec(kind=model, fidelity=1.0 if u["model"] == "e2e_fid" else 0.0)
    spec = RunSpec(protocol=u["protocol"], model=model, loss=loss, model_seed=int(u["model_seed"]),
                   start_day=int(u["start_day"]), val_days=u.get("val_days"))
    res = backtest(panel, spec)
    h = panel.cfg.horizon
    summ = {"all": summarize(res, h), "overlap": summarize(res, h, from_day=2000)}
    return {"unit": u, "env": _env(), "dgp": panel.cfg.as_dict(), "summary": summ, "fits": res["fits"],
            "decisions": res["decisions"], "sec": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    units = grid_units(a.grid)
    if a.list:
        print(f"{a.grid}: {len(units)} units")
        return
    out = ROOT / "results" / a.grid
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, u in enumerate(units):
        if i % a.n_chunks != a.chunk:
            continue
        path = out / f"{i:05d}.json"
        if path.exists():
            continue
        r = run_unit(u)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, default=float), encoding="utf-8")
        tmp.replace(path)
        s = r["summary"].get("all", r["summary"])
        print(f"[{a.grid} {i}] {json.dumps(u)} sec={r['sec']} "
              + " ".join(f"{k}={s[k]:.4g}" for k in ("true_ic", "ann_net", "chosen_epoch") if k in s), flush=True)
        done += 1
        if a.limit is not None and done >= a.limit:
            break


if __name__ == "__main__":
    main()
