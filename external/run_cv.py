# -*- coding: utf-8 -*-
"""T19: choose the number of training epochs of Qlib LSTM / ALSTM by purged K-fold CV, never looking at the test period.

One job per process (`--job`):
  fold0 .. fold4  2008-01-01 .. 2016-12-31 (the official train + valid segments) is cut into 5 contiguous folds of equal
                  numbers of trading days. Fold k is the validation set; the training set is everything else in
                  2008-2016 minus an embargo of 20 trading days on each side of the fold (samples dated there are
                  dropped). A freshly initialized official model is trained for E_MAX = 20 epochs with no early stop,
                  and the validation score of every epoch is recorded (Qlib's own score of the model: -MSE on the fold).
                  -> ~/ext/runs/<m>_cvfold/seed_XX_fK.json
  full            e* = argmax over epochs of the 5-fold mean validation curve (ties -> fewer epochs). A freshly
                  initialized official model is trained on all of 2008-01-01 .. 2016-12-31 for e* epochs and the last
                  (e*-th) epoch is deployed; official test period and backtest. -> ~/ext/runs/<m>_cv/seed_XX.json
  ctrl            control: e* epochs on the official training segment only (2008-01-01 .. 2014-12-31, official
                  valid segment logged only), i.e. the official run with a fixed budget of e*. -> <m>_cvctrl/seed_XX.json

Everything else is the official Alpha158 config: model kwargs (constant lr, Adam), batch, loss, features. Training
uses Qlib's own fit() unchanged (fixed_budget.FixedBudget* for full/ctrl, which deploys the last epoch; Qlib's
LSTM/ALSTM with early_stop > E_MAX for the folds); for a fold the dataset handed to fit() only filters the samples
of the training segment, so batches, shuffling and per-epoch evaluation are Qlib's. Feature normalization is fitted
on 2008-2016 for the folds and full (the data the rule may see) and on 2008-2014 for ctrl (as the official config).

    ~/ext/.venv-qlib/bin/python external/run_cv.py --model lstm --seed 0 --job fold0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from run_seeds import CONFIGS, RUNS, ensure_experiment, metrics, seeded_model_config  # noqa: E402

N_FOLDS, E_MAX, EMBARGO = 5, 20, 20
CV_START, CV_END = "2008-01-01", "2016-12-31"
OFFICIAL_TRAIN_END = "2014-12-31"
TEST = ("2017-01-01", "2020-08-01")
FB_CLASS = {"lstm": "FixedBudgetLSTM", "alstm": "FixedBudgetALSTM"}
SUFFIX = os.environ.get("T19_SUFFIX", "")                 # smoke tests only


def run_dir(model: str, kind: str) -> Path:
    return RUNS / f"{model}_{kind}{SUFFIX}"


def fold_bounds(cal: list, k: int) -> tuple:
    """(valid start, valid end, first excluded date, last excluded date) of fold k over the trading calendar `cal`."""
    parts = np.array_split(np.arange(len(cal)), N_FOLDS)
    a, b = int(parts[k][0]), int(parts[k][-1])
    return cal[a], cal[b], cal[max(0, a - EMBARGO)], cal[min(len(cal) - 1, b + EMBARGO)]


class _Subset:
    """the training TSDataSampler restricted to some of its samples (all Qlib's fit() uses of it)."""

    def __init__(self, base, pos):
        self.base, self.pos = base, np.asarray(pos)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        return self.base[int(self.pos[i])]

    @property
    def empty(self):
        return len(self) == 0

    def config(self, **kwargs):
        self.base.config(**kwargs)

    def get_index(self):
        return self.base.get_index()[self.pos]


class _PurgedTrain:
    """dataset proxy for Qlib's fit(): prepare('train') drops the samples dated in [lo, hi] (validation fold + embargo)."""

    def __init__(self, dataset, lo, hi):
        self.dataset, self.lo, self.hi = dataset, pd.Timestamp(lo), pd.Timestamp(hi)
        self.n_train = self.n_dropped = None

    def prepare(self, segments, *args, **kwargs):
        d = self.dataset.prepare(segments, *args, **kwargs)
        if segments != "train":
            return d
        dates = d.get_index().get_level_values("datetime")
        keep = np.where((dates < self.lo) | (dates > self.hi))[0]
        self.n_train, self.n_dropped = int(len(keep)), int(len(dates) - len(keep))
        return _Subset(d, keep)


def build_dataset(task: dict, train: tuple, valid: tuple, fit_end: str):
    from qlib.utils import init_instance_by_config
    ds_cfg = copy.deepcopy(task["dataset"])
    h = ds_cfg["kwargs"]["handler"]["kwargs"]
    h.update(fit_start_time=CV_START, fit_end_time=fit_end)
    ds_cfg["kwargs"]["segments"] = {"train": list(train), "valid": list(valid), "test": list(TEST)}
    return init_instance_by_config(ds_cfg)


def chosen_epoch(model: str, tag: str) -> dict:
    curves = []
    for k in range(N_FOLDS):
        f = run_dir(model, "cvfold") / f"seed_{tag}_f{k}.json"
        if not f.exists():
            raise SystemExit(f"fold {k} of {model} seed {tag} not done yet")
        curves.append(json.loads(f.read_text(encoding="utf-8"))["valid_curve"])
    c = np.array(curves)
    mean = c.mean(axis=0)
    return {"e_star": int(np.argmax(mean)) + 1, "mean_curve": mean.tolist(), "fold_curves": c.tolist(),
            "fold_best": [int(np.argmax(r)) + 1 for r in c]}


def run_fold(task, model_cfg, model, tag, k):
    from qlib.data import D
    from qlib.utils import init_instance_by_config
    cal = [pd.Timestamp(x) for x in D.calendar(start_time=CV_START, end_time=CV_END)]
    vs, ve, lo, hi = fold_bounds(cal, k)
    t0 = time.time()
    dataset = build_dataset(task, (CV_START, CV_END), (str(vs.date()), str(ve.date())), CV_END)
    t_data = time.time() - t0
    cfg = copy.deepcopy(model_cfg)
    cfg["kwargs"].update(n_epochs=E_MAX, early_stop=E_MAX + 1000)
    m = init_instance_by_config(cfg)
    proxy = _PurgedTrain(dataset, lo, hi)
    evals = {}
    m.fit(proxy, evals_result=evals)
    valid = [float(v) for v in evals["valid"]]
    out = {"model": model, "seed": int(tag), "fold": k, "valid_start": str(vs.date()), "valid_end": str(ve.date()),
           "excluded_from_train": [str(lo.date()), str(hi.date())], "n_train": proxy.n_train,
           "n_dropped": proxy.n_dropped, "valid_curve": valid, "train_curve": [float(v) for v in evals.get("train", [])],
           "best_epoch": int(np.argmax(valid)) + 1, "data_sec": round(t_data, 1),
           "fit_sec": round(time.time() - t0 - t_data, 1)}
    d = run_dir(model, "cvfold")
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"seed_{tag}_f{k}.json.tmp"
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(d / f"seed_{tag}_f{k}.json")
    print(f"[{model} seed {tag} fold {k}] valid {vs.date()}..{ve.date()} train n={proxy.n_train} "
          f"(dropped {proxy.n_dropped}) best={out['best_epoch']} sec={time.time() - t0:.0f}", flush=True)


def run_deploy(task, model_cfg, model, tag, job):
    from qlib.utils import fill_placeholder, init_instance_by_config
    from qlib.workflow import R
    sel = chosen_epoch(model, tag)
    e = sel["e_star"]
    if job == "full":
        name = f"{model}_cv{SUFFIX}"
        train, valid, fit_end = (CV_START, CV_END), ("2016-12-01", CV_END), CV_END   # valid: last month, logged only
    else:
        name = f"{model}_cvctrl{SUFFIX}"
        train, valid, fit_end = (CV_START, OFFICIAL_TRAIN_END), ("2015-01-01", CV_END), OFFICIAL_TRAIN_END
    t0 = time.time()
    dataset = build_dataset(task, train, valid, fit_end)
    cfg = copy.deepcopy(model_cfg)
    cfg["class"], cfg["module_path"] = FB_CLASS[model], "fixed_budget"
    cfg["kwargs"].update(n_epochs=e, early_stop=e + 1000)
    m = init_instance_by_config(cfg)
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_experiment(R, f"{name}_seeds")
    fitted = False
    for attempt in range(3):      # MLflow's file store can read back a just-logged metric file as empty ("malformed")
        try:
            with R.start(experiment_name=f"{name}_seeds", recorder_name=f"seed_{tag}" + (f"_retry{attempt}" if attempt else "")):
                rec = R.get_recorder()
                if not fitted:
                    m.fit(dataset)
                    fitted, t_fit = True, time.time() - t0
                records = fill_placeholder(copy.deepcopy(task["record"]), {"<MODEL>": m, "<DATASET>": dataset})
                for r in records:
                    init_instance_by_config(r, recorder=rec, default_module="qlib.workflow.record_temp",
                                            try_kwargs={"model": m, "dataset": dataset}).generate()
                res = metrics(rec)
                pred = rec.load_object("pred.pkl")
            break
        except Exception as ex:   # ValueError, or Qlib's LoadObjectError wrapping it
            if "malformed" not in str(ex) or attempt == 2:
                raise
            print(f"[{name} seed {tag}] MLflow metric read failed ({ex}); retrying in a new recorder", flush=True)
            time.sleep(10)
    pred.to_pickle(out_dir / f"seed_{tag}_pred.pkl")
    res.update(model=name, seed=int(tag), job=job, config=str(CONFIGS[model]), model_cfg=cfg, train=list(train),
               fit_end=fit_end, deployed_epoch=e, **sel, fit_sec=round(t_fit, 1), total_sec=round(time.time() - t0, 1))
    tmp = out_dir / f"seed_{tag}.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    tmp.replace(out_dir / f"seed_{tag}.json")
    print(f"[{name} seed {tag}] e*={e} folds best={sel['fold_best']} IC={res['IC']:.4f} RankIC={res['Rank IC']:.4f} "
          f"exc_w_cost={res['ann_excess_w_cost']:.4f} IR={res['ir_w_cost']:.3f} sec={res['total_sec']}", flush=True)


def output_path(model: str, seed: int, job: str) -> Path:
    tag = f"{seed:02d}"
    if job.startswith("fold"):
        return run_dir(model, "cvfold") / f"seed_{tag}_f{int(job[4:])}.json"
    return run_dir(model, "cv" if job == "full" else "cvctrl") / f"seed_{tag}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "alstm"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--job", choices=[f"fold{k}" for k in range(N_FOLDS)] + ["full", "ctrl"], required=True)
    a = ap.parse_args()
    if output_path(a.model, a.seed, a.job).exists():
        print("done already")
        return
    os.chdir(RUNS)

    import qlib
    from qlib.cli.run import render_template
    from ruamel.yaml import YAML
    cfg = YAML(typ="safe", pure=True).load(render_template(str(CONFIGS[a.model])))
    qlib.init(**cfg["qlib_init"])
    task = cfg["task"]
    model_cfg = seeded_model_config(task["model"], a.seed)
    if os.environ.get("T19_EMAX"):                           # smoke tests only
        global E_MAX
        E_MAX = int(os.environ["T19_EMAX"])
    tag = f"{a.seed:02d}"
    if a.job.startswith("fold"):
        run_fold(task, model_cfg, a.model, tag, int(a.job[4:]))
    else:
        run_deploy(task, model_cfg, a.model, tag, a.job)


if __name__ == "__main__":
    main()
