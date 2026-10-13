# -*- coding: utf-8 -*-
"""P7 external screening experiment, second model: the official Qlib MLP (pre-registered in external/PREREG_P7_screen_addendum_zh.md).

Generated from p7_screen.py by exact replacements: same design, adoption rule and seeds; model MLP (DNNModelPytorch),
candidate changes in ROUNDS below, run directories p7m_<name>, adoption file p7_screen_mlp_adopted.json. The thread count
is fixed with torch.set_num_threads (DNNModelPytorch has no num_threads argument). Every override key is checked against
the model signature before any run.


A miniature development process on the official LightGBM / Alpha158 / CSI 300 benchmark. Three rounds of six candidate
changes; after each round the configuration with the highest mean annualized excess return (after costs) on seeds 0, 1, 2
is adopted and becomes the baseline of the next round, exactly as in the internal development records of the paper.
Every configuration is run on seeds 0-14, so that each round-2 and round-3 screen can be estimated twice: on the
development seeds 0-2 (the seeds on which its baseline was adopted) and on the fresh seeds 12-14; seeds 3-11 are the
common expansion seeds.

(LightGBM version:) The only deviation from the official configuration is num_threads (fixed at THREADS for every run, including the base,
so that thread count cannot differ between configurations).

Usage (Qlib environment, never the repo's .venv):
    ~/ext/.venv-qlib/bin/python external/p7_screen.py --round 0                      # base configuration
    ~/ext/.venv-qlib/bin/python external/p7_screen.py --round 1 [--only r1_lr010,r1_lr030]
    ~/ext/.venv-qlib/bin/python external/p7_screen.py --adopt 1                      # writes results/external/p7_screen_adopted.json
    ~/ext/.venv-qlib/bin/python external/p7_screen.py --round 2 ...; --adopt 2; --round 3 ...
    ~/ext/.venv-qlib/bin/python external/p7_screen.py --smoke                        # base, seed 0 only, timing
Output per run: ~/ext/runs/p7s_<name>/seed_XX.json (metrics as in run_seeds.metrics, plus the overrides used).
Runs whose json exists are skipped, so the script can be restarted.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_seeds as RS  # noqa: E402  (CONFIGS, RUNS, metrics, ensure_experiment)

THREADS = 4
SEEDS = list(range(15))
DEV, FRESH, EXP = [0, 1, 2], [12, 13, 14], list(range(3, 12))
ROUNDS = {
    1: {"m1_lr001": {"lr": 0.001}, "m1_lr004": {"lr": 0.004}, "m1_wd0": {"weight_decay": 0.0},
        "m1_wd001": {"weight_decay": 0.001}, "m1_bs4096": {"batch_size": 4096}, "m1_es25": {"early_stop_rounds": 25}},
    2: {"m2_lr003": {"lr": 0.003}, "m2_wd0005": {"weight_decay": 0.0005}, "m2_bs16384": {"batch_size": 16384},
        "m2_es100": {"early_stop_rounds": 100}, "m2_ev10": {"eval_steps": 10}, "m2_steps2000": {"max_steps": 2000}},
    3: {"m3_lr0015": {"lr": 0.0015}, "m3_lr0025": {"lr": 0.0025}, "m3_wd0001": {"weight_decay": 0.0001},
        "m3_bs6144": {"batch_size": 6144}, "m3_es75": {"early_stop_rounds": 75}, "m3_ev40": {"eval_steps": 40}},
}
ADOPT = HERE.parent / "results" / "external" / "p7_screen_mlp_adopted.json"
RUNS = RS.RUNS


def adopted() -> dict:
    """round -> {"name": run directory suffix, "overrides": dict}; round 0 is the base configuration."""
    a = {"0": {"name": "base", "overrides": {}}}
    if ADOPT.exists():
        a.update(json.loads(ADOPT.read_text(encoding="utf-8")))
    return a


def configs_of_round(r: int) -> dict:
    """name -> overrides (relative to the official configuration) for the configurations run in round r."""
    if r == 0:
        return {"base": {}}
    a = adopted()
    if str(r - 1) not in a:
        sys.exit(f"round {r - 1} has not been adopted yet: run --adopt {r - 1} first")
    prev = a[str(r - 1)]["overrides"]
    out = {}
    for name, ch in ROUNDS[r].items():
        ov = {**prev, **ch}
        if ov == prev:
            print(f"skip {name}: identical to the adopted configuration of round {r - 1}")
            continue
        out[name] = ov
    return out


def load_metric(name: str, seed: int, key: str = "ann_excess_w_cost"):
    f = RUNS / f"p7m_{name}" / f"seed_{seed:02d}.json"
    return json.loads(f.read_text(encoding="utf-8"))[key] if f.exists() else None


def adopt(r: int) -> None:
    """adopt the configuration with the highest mean over seeds 0-2 among the previous adopted one and round r's candidates."""
    a = adopted()
    prev = a[str(r - 1)]
    pool = {prev["name"]: prev["overrides"], **configs_of_round(r)}
    scores = {}
    for name in pool:
        v = [load_metric(name, s) for s in DEV]
        if any(x is None for x in v):
            sys.exit(f"{name} is missing seeds {DEV}: finish round {r} first")
        scores[name] = sum(v) / len(v)
    win = max(scores, key=scores.get)
    a[str(r)] = {"name": win, "overrides": pool[win], "dev_means": scores}
    ADOPT.parent.mkdir(parents=True, exist_ok=True)
    ADOPT.write_text(json.dumps({k: v for k, v in a.items() if k != "0"}, indent=1), encoding="utf-8")
    print(f"round {r}: adopted {win} (mean over seeds {DEV} = {scores[win]:.4f}); all: " +
          ", ".join(f"{k} {v:.4f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])))


def run(todo: dict, seeds: list, max_runs: int = 0) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    jobs = [(n, ov, s) for n, ov in todo.items() for s in seeds if not (RUNS / f"p7m_{n}" / f"seed_{s:02d}.json").exists()]
    if not jobs:
        print("all runs done")
        return
    if max_runs:
        jobs = jobs[:max_runs]   # memory grows by ~0.8 GB per run within one process; restart to free it
    RUNS.mkdir(parents=True, exist_ok=True)
    os.chdir(RUNS)
    import qlib
    from qlib.cli.run import render_template
    from qlib.utils import fill_placeholder, init_instance_by_config
    from qlib.workflow import R
    from ruamel.yaml import YAML
    cfg = YAML(typ="safe", pure=True).load(render_template(str(RS.CONFIGS["mlp"])))
    import inspect
    import torch
    torch.set_num_threads(THREADS)
    from qlib.contrib.model.pytorch_nn import DNNModelPytorch
    allowed = set(inspect.signature(DNNModelPytorch.__init__).parameters)
    bad = sorted({k for _, ov, _ in jobs for k in ov} - allowed)
    if bad:
        sys.exit(f"override keys not accepted by DNNModelPytorch: {bad}")
    qlib.init(**cfg["qlib_init"])
    task = cfg["task"]
    t0 = time.time()
    dataset = init_instance_by_config(task["dataset"])
    print(f"dataset ready in {time.time() - t0:.0f}s; {len(jobs)} runs", flush=True)
    for name, ov, seed in jobs:
        t1 = time.time()
        mcfg = copy.deepcopy(task["model"])
        mcfg["kwargs"].update({**ov, "seed": int(seed)})
        model = init_instance_by_config(mcfg)
        exp = f"p7m_{name}"
        RS.ensure_experiment(R, exp)
        with R.start(experiment_name=exp, recorder_name=f"seed_{seed:02d}"):
            rec = R.get_recorder()
            model.fit(dataset)
            records = fill_placeholder(copy.deepcopy(task["record"]), {"<MODEL>": model, "<DATASET>": dataset})
            for rr in records:
                init_instance_by_config(rr, recorder=rec, default_module="qlib.workflow.record_temp",
                                        try_kwargs={"model": model, "dataset": dataset}).generate()
            res = RS.metrics(rec)
        res.update(name=name, seed=seed, overrides=ov, threads=THREADS, model_cfg=mcfg, total_sec=round(time.time() - t1, 1))
        d = RUNS / exp
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"seed_{seed:02d}.json.tmp"
        tmp.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        tmp.replace(d / f"seed_{seed:02d}.json")
        print(f"[{name} seed {seed:02d}] IC={res['IC']:.4f} exc_w_cost={res['ann_excess_w_cost']:.4f} sec={res['total_sec']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--round", type=int, choices=[0, 1, 2, 3])
    g.add_argument("--adopt", type=int, choices=[1, 2, 3])
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated configuration names of the round")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many runs (0 = all); rerun to continue")
    a = ap.parse_args()
    if a.adopt:
        adopt(a.adopt)
    elif a.smoke:
        run({"base": {}}, [0])
    else:
        todo = configs_of_round(a.round)
        if a.only:
            keep = set(a.only.split(","))
            todo = {k: v for k, v in todo.items() if k in keep}
        run(todo, SEEDS, a.max_runs)


if __name__ == "__main__":
    main()
