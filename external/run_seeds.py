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


BENCH = Path.home() / "ext" / "qlib" / "examples" / "benchmarks"
CONFIGS = {"lgb": BENCH / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml",
           "mlp": BENCH / "MLP" / "workflow_config_mlp_Alpha158.yaml",
           "gru": BENCH / "GRU" / "workflow_config_gru_Alpha158.yaml",
           "lstm": BENCH / "LSTM" / "workflow_config_lstm_Alpha158.yaml",
           "alstm": BENCH / "ALSTM" / "workflow_config_alstm_Alpha158.yaml"}
# T17 / T18-A fixed-budget variants: same official config, only n_epochs and early_stop (above n_epochs, so it never
# triggers), and the model class swapped for fixed_budget.py's wrapper that deploys the last epoch (Qlib reloads the
# best-validation epoch otherwise). name -> (base, class, n_epochs, extra deployed epochs). Training up to epoch e does
# not depend on n_epochs, so the 10-epoch run also writes the 5-epoch result (as <base>_fb5).
VARIANTS = {"lstm_fb": ("lstm", "FixedBudgetLSTM", 20, ()), "alstm_fb": ("alstm", "FixedBudgetALSTM", 20, ()),
            "lstm_fb10": ("lstm", "FixedBudgetLSTM", 10, (5,)), "alstm_fb10": ("alstm", "FixedBudgetALSTM", 10, (5,))}
FB_EARLY_STOP = 1000
for _v, (_base, *_rest) in VARIANTS.items():
    CONFIGS[_v] = CONFIGS[_base]
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
        cfg["kwargs"]["seed"] = int(seed)          # LGBModel passes it to lightgbm params; the pytorch models take `seed`
    return cfg


def ensure_experiment(R, name: str) -> None:
    """create the MLflow experiment up front; two processes creating it at once raise ExpAlreadyExistError (T17)."""
    for attempt in range(3):
        try:
            R.get_exp(experiment_name=name, create=True)
            return
        except Exception:
            time.sleep(5 * (attempt + 1))
    R.get_exp(experiment_name=name, create=False)


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

    # parse exactly as qrun does (jinja render + ruamel YAML 1.2): PyYAML reads "lr: 2e-4" as a string
    from qlib.cli.run import render_template
    from ruamel.yaml import YAML
    cfg = YAML(typ="safe", pure=True).load(render_template(str(CONFIGS[a.model])))
    qlib.init(**cfg["qlib_init"])
    task = cfg["task"]
    deploys = [(a.model, None)]                        # (output name, epoch to deploy; None = as trained)
    if a.model in VARIANTS:
        base, cls, n_epochs, extra = VARIANTS[a.model]
        task["model"]["class"] = cls
        task["model"]["module_path"] = "fixed_budget"     # external/fixed_budget.py (this script's directory is on sys.path)
        task["model"]["kwargs"].update(n_epochs=n_epochs, early_stop=FB_EARLY_STOP)
        deploys += [(f"{base}_fb{e}", e) for e in extra]
    t0 = time.time()
    dataset = init_instance_by_config(task["dataset"])
    t_data = time.time() - t0
    print(f"dataset ready in {t_data:.0f}s", flush=True)

    for seed in seeds:
        tag = "none" if seed is None else f"{seed:02d}"
        t1 = time.time()
        model = init_instance_by_config(seeded_model_config(task["model"], seed))
        for i, (name, epoch) in enumerate(deploys):
            dname = RUNS / name
            dname.mkdir(parents=True, exist_ok=True)
            ensure_experiment(R, f"{name}_seeds")
            with R.start(experiment_name=f"{name}_seeds", recorder_name=f"seed_{tag}"):
                rec = R.get_recorder()
                if i == 0:                                     # train inside the first recorder (MLP logs metrics to it)
                    model.fit(dataset)
                    t_fit = time.time() - t1
                if epoch is not None:
                    model.deploy_epoch(epoch)
                records = fill_placeholder(copy.deepcopy(task["record"]), {"<MODEL>": model, "<DATASET>": dataset})
                for r in records:
                    init_instance_by_config(r, recorder=rec, default_module="qlib.workflow.record_temp",
                                            try_kwargs={"model": model, "dataset": dataset}).generate()
                res = metrics(rec)
                pred = rec.load_object("pred.pkl")
            pred.to_pickle(dname / f"seed_{tag}_pred.pkl")
            mcfg = copy.deepcopy(task["model"])
            if epoch is not None:
                mcfg["deployed_epoch"] = epoch
            res.update(model=name, seed=seed, config=str(CONFIGS[a.model]), model_cfg=mcfg, dataset_sec=round(t_data, 1),
                       fit_sec=round(t_fit, 1), total_sec=round(time.time() - t1, 1))
            tmp = dname / f"seed_{tag}.json.tmp"
            tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
            tmp.replace(dname / f"seed_{tag}.json")
            print(f"[{name} seed {tag}] IC={res['IC']:.4f} RankIC={res['Rank IC']:.4f} "
                  f"exc_w_cost={res['ann_excess_w_cost']:.4f} IR={res['ir_w_cost']:.3f} sec={res['total_sec']}", flush=True)


if __name__ == "__main__":
    main()
