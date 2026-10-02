# -*- coding: utf-8 -*-
"""Run an official Qlib benchmark config over several random seeds (only the seed changes).

The dataset is built once per process; each seed then goes through the same steps as `qrun`
(model.fit -> SignalRecord -> SigAnaRecord -> PortAnaRecord, via qlib's own record classes).
Per seed it writes, outside the repo, ~/ext/runs/<name>/seed_XX.json (overall and per-year metrics)
and seed_XX_pred.pkl (daily test-period scores). Seeds whose json exists are skipped.

    ~/ext/.venv-qlib/bin/python external/run_seeds.py --model lgb --seeds 0-19
    ~/ext/.venv-qlib/bin/python external/run_seeds.py --model lgb --seeds none   # official config as is (parity check vs qrun)

Run with the separate Qlib environment (~/ext/.venv-qlib), never the repo's .venv.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")   # MLflow 3.x refuses the file store otherwise

import yaml

BENCH = Path.home() / "ext" / "qlib" / "examples" / "benchmarks"
CONFIGS = {"lgb": BENCH / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml",
           "mlp": BENCH / "MLP" / "workflow_config_mlp_Alpha158.yaml"}
RUNS = Path.home() / "ext" / "runs"
YEARS = (2017, 2018, 2019, 2020)


def parse_seeds(s: str) -> list:
    if s == "none":
        return [None]
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def seeded_model_config(model_cfg: dict, seed) -> dict:
    cfg = copy.deepcopy(model_cfg)
    if seed is not None:
        cfg["kwargs"]["seed"] = int(seed)          # LGBModel passes it to lightgbm params; DNNModelPytorch takes `seed`
    return cfg


def metrics(rec) -> dict:
    from qlib.contrib.evaluate import risk_analysis
    ic = rec.load_object("sig_analysis/ic.pkl")
    ric = rec.load_object("sig_analysis/ric.pkl")
    rep = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
    ex0 = rep["return"] - rep["bench"]
    ex1 = ex0 - rep["cost"]
    ra0, ra1 = risk_analysis(ex0, freq="day")["risk"], risk_analysis(ex1, freq="day")["risk"]
    out = {"IC": ic.mean(), "ICIR": ic.mean() / ic.std(), "Rank IC": ric.mean(), "Rank ICIR": ric.mean() / ric.std(),
           "ann_excess_wo_cost": ra0["annualized_return"], "ir_wo_cost": ra0["information_ratio"],
           "ann_excess_w_cost": ra1["annualized_return"], "ir_w_cost": ra1["information_ratio"],
           "mdd_w_cost": ra1["max_drawdown"], "years": {}}
    for y in YEARS:
        e0, e1 = ex0[ex0.index.year == y], ex1[ex1.index.year == y]
        out["years"][str(y)] = {
            "IC": ic[ic.index.year == y].mean(), "Rank IC": ric[ric.index.year == y].mean(),
            "ann_excess_w_cost": risk_analysis(e1, freq="day")["risk"]["annualized_return"],
            "ann_excess_wo_cost": risk_analysis(e0, freq="day")["risk"]["annualized_return"],
            "n_days": int(len(e1))}
    return json.loads(json.dumps(out, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 0-19, 0-9, 3,5 or none")
    a = ap.parse_args()

    out_dir = RUNS / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in parse_seeds(a.seeds)
             if not (out_dir / f"seed_{'none' if s is None else f'{s:02d}'}.json").exists()]
    if not seeds:
        print("all seeds done")
        return
    os.chdir(RUNS)                                   # mlruns/ goes to ~/ext/runs, outside the repo

    import qlib
    from qlib.utils import fill_placeholder, init_instance_by_config
    from qlib.workflow import R

    cfg = yaml.safe_load(CONFIGS[a.model].read_text())
    qlib.init(**cfg["qlib_init"])
    task = cfg["task"]
    t0 = time.time()
    dataset = init_instance_by_config(task["dataset"])
    t_data = time.time() - t0
    print(f"dataset ready in {t_data:.0f}s", flush=True)

    for seed in seeds:
        tag = "none" if seed is None else f"{seed:02d}"
        t1 = time.time()
        with R.start(experiment_name=f"{a.model}_seeds", recorder_name=f"seed_{tag}"):
            rec = R.get_recorder()
            model = init_instance_by_config(seeded_model_config(task["model"], seed))
            model.fit(dataset)
            t_fit = time.time() - t1
            records = fill_placeholder(copy.deepcopy(task["record"]), {"<MODEL>": model, "<DATASET>": dataset})
            for r in records:
                init_instance_by_config(r, recorder=rec, default_module="qlib.workflow.record_temp",
                                        try_kwargs={"model": model, "dataset": dataset}).generate()
            res = metrics(rec)
            pred = rec.load_object("pred.pkl")
        pred.to_pickle(out_dir / f"seed_{tag}_pred.pkl")
        res.update(model=a.model, seed=seed, config=str(CONFIGS[a.model]), dataset_sec=round(t_data, 1),
                   fit_sec=round(t_fit, 1), total_sec=round(time.time() - t1, 1))
        tmp = out_dir / f"seed_{tag}.json.tmp"
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        tmp.replace(out_dir / f"seed_{tag}.json")
        print(f"[{a.model} seed {tag}] IC={res['IC']:.4f} RankIC={res['Rank IC']:.4f} "
              f"exc_w_cost={res['ann_excess_w_cost']:.4f} IR={res['ir_w_cost']:.3f} sec={res['total_sec']}", flush=True)


if __name__ == "__main__":
    main()
