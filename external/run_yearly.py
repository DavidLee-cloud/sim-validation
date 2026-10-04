# -*- coding: utf-8 -*-
"""T18-B / T18-C: yearly retraining on an expanding window for Qlib LSTM / ALSTM (one seed per process).

For each test year Y in 2017-2020 a new dataset and a freshly initialized model are built from the official
Alpha158 config; the model predicts year Y only (2020: to 2020-08-01). The four yearly predictions are concatenated
and go through Qlib's own SigAnaRecord and PortAnaRecord (same backtest and costs as the official config), so the
output (~/ext/runs/<name>/seed_XX.json and _pred.pkl) has the same format as run_seeds.py.

  <m>_proto / <m>_proto10 (B20 / B10, full fixed-budget protocol, fixed_budget.Protocol*): train 2008-01-01 ..
      (Y-1)-12-31; no validation set for model selection (the valid segment is the last month of the training
      segment, logged only); 20 / 10 epochs, deploy the last; AdamW 0.03; cosine lr; recency-weighted sampling, half-life 504 trading days.
  <m>_esyr (C, official early stopping, retrained yearly): official model class and kwargs; train
      2008-01-01 .. (Y-3)-12-31, valid (Y-2)-01-01 .. (Y-1)-12-31 (two years, like the official split).
In both, the feature normalization is fitted on the training segment (fit_end_time = end of the training segment),
as the official config fits it on its training segment. For C, year 2017 has exactly the official split, so its 2017
predictions are compared with the official early-stopping run of the same seed (stored as check_2017_max_abs_diff).

    ~/ext/.venv-qlib/bin/python external/run_yearly.py --model lstm_proto --seed 0
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from run_seeds import CONFIGS, RUNS, ensure_experiment, metrics, seeded_model_config  # noqa: E402

# name -> (base, arm, protocol epochs). B20 (<m>_proto) was stopped after seeds 0-4 (2026-10-04): A showed 20 bare epochs
# over-train in Qlib and the first B20 seeds agreed. B10 (<m>_proto10) uses 10 epochs; 10 rather than A's best 5 so as
# not to pick the epoch count on the test period (10 was still informed by A's epoch curve).
ARMS = {"lstm_proto": ("lstm", "B", 20), "alstm_proto": ("alstm", "B", 20),
        "lstm_proto10": ("lstm", "B", 10), "alstm_proto10": ("alstm", "B", 10),
        "lstm_esyr": ("lstm", "C", None), "alstm_esyr": ("alstm", "C", None)}
PROTO_CLASS = {"lstm": "ProtocolLSTM", "alstm": "ProtocolALSTM"}
PROTO_KW = dict(adamw_weight_decay=0.03, half_life=504.0)
YEARS = (2017, 2018, 2019, 2020)
DATA_START, DATA_END = "2008-01-01", "2020-08-01"


def year_split(arm: str, Y: int) -> dict:
    test = (f"{Y}-01-01", min(f"{Y}-12-31", DATA_END))
    if arm == "B":
        train = (DATA_START, f"{Y - 1}-12-31")
        valid = (f"{Y - 1}-12-01", f"{Y - 1}-12-31")       # last month of the training segment, logged only
    else:
        train = (DATA_START, f"{Y - 3}-12-31")
        valid = (f"{Y - 2}-01-01", f"{Y - 1}-12-31")
    return {"train": train, "valid": valid, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    base, arm, epochs = ARMS[a.model]
    years = tuple(int(y) for y in os.environ.get("T18_YEARS", ",".join(map(str, YEARS))).split(","))  # smoke tests only
    name = a.model + os.environ.get("T18_SUFFIX", "")
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.seed:02d}"
    if (out_dir / f"seed_{tag}.json").exists():
        print("done already")
        return
    os.chdir(RUNS)

    import qlib
    from qlib.cli.run import render_template
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord
    from ruamel.yaml import YAML

    cfg = YAML(typ="safe", pure=True).load(render_template(str(CONFIGS[base])))
    qlib.init(**cfg["qlib_init"])
    task = cfg["task"]
    model_cfg = copy.deepcopy(task["model"])
    if arm == "B":
        model_cfg["class"], model_cfg["module_path"] = PROTO_CLASS[base], "fixed_budget"
        model_cfg["kwargs"].update(PROTO_KW, n_epochs=epochs)
        if os.environ.get("T18_EPOCHS"):                  # smoke tests only
            model_cfg["kwargs"]["n_epochs"] = int(os.environ["T18_EPOCHS"])
    model_cfg = seeded_model_config(model_cfg, a.seed)

    t1 = time.time()
    preds, labels, info = [], [], {}
    stash = {k: out_dir / f"seed_{tag}_{k}" for k in ("pred.pkl", "label.pkl", "info.json")}   # trained, not yet recorded
    if all(f.exists() for f in stash.values()):
        print(f"[{name} seed {tag}] resuming from saved predictions (training already done)", flush=True)
        preds, labels = [pd.read_pickle(stash["pred.pkl"])], [pd.read_pickle(stash["label.pkl"])]
        info = json.loads(stash["info.json"].read_text(encoding="utf-8"))
        years = ()
    for Y in years:
        ty = time.time()
        split = year_split(arm, Y)
        ds_cfg = copy.deepcopy(task["dataset"])
        h = ds_cfg["kwargs"]["handler"]["kwargs"]
        h.update(start_time=DATA_START, end_time=DATA_END, fit_start_time=DATA_START, fit_end_time=split["train"][1])
        ds_cfg["kwargs"]["segments"] = {k: list(v) for k, v in split.items()}
        dataset = init_instance_by_config(ds_cfg)
        t_data = time.time() - ty
        model = init_instance_by_config(model_cfg)
        evals = {}
        model.fit(dataset, evals_result=evals)
        p = model.predict(dataset)
        p = p.to_frame("score") if isinstance(p, pd.Series) else p
        preds.append(p)
        labels.append(SignalRecord.generate_label(dataset))
        valid = [float(v) for v in evals.get("valid", [])]
        info[str(Y)] = {"split": split, "data_sec": round(t_data, 1), "fit_sec": round(time.time() - ty - t_data, 1),
                        "valid_curve": valid, "epochs_trained": len(valid),
                        "best_epoch": (int(max(range(len(valid)), key=valid.__getitem__)) + 1) if valid else None}
        print(f"[{name} seed {tag} year {Y}] epochs={len(valid)} best={info[str(Y)]['best_epoch']} "
              f"sec={time.time() - ty:.0f}", flush=True)
        del dataset, model
        gc.collect()

    pred = pd.concat(preds).sort_index()
    label = pd.concat(labels).sort_index()
    if not stash["info.json"].exists():
        pred.to_pickle(stash["pred.pkl"])
        label.to_pickle(stash["label.pkl"])
        stash["info.json"].write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    check = None
    if arm == "C":                                        # 2017 uses exactly the official split
        ref = RUNS / base / f"seed_{tag}_pred.pkl"
        if ref.exists():
            r = pd.read_pickle(ref)["score"]
            mine = pred["score"]
            idx = mine.index[mine.index.get_level_values("datetime").year == 2017]
            check = float((mine.loc[idx] - r.reindex(idx)).abs().max())
    ensure_experiment(R, f"{name}_seeds")
    for attempt in range(3):      # MLflow's file store can read back a just-logged metric file as empty ("malformed")
        try:
            with R.start(experiment_name=f"{name}_seeds", recorder_name=f"seed_{tag}" + (f"_retry{attempt}" if attempt else "")):
                rec = R.get_recorder()
                rec.save_objects(**{"pred.pkl": pred, "label.pkl": label})
                for r in task["record"]:
                    if r["class"] == "SignalRecord":
                        continue
                    init_instance_by_config(r, recorder=rec, default_module="qlib.workflow.record_temp").generate()
                res = metrics(rec)
            break
        except ValueError as e:
            if "malformed" not in str(e) or attempt == 2:
                raise
            print(f"[{name} seed {tag}] MLflow metric read failed ({e}); retrying in a new recorder", flush=True)
            time.sleep(10)
    pred.to_pickle(out_dir / f"seed_{tag}_pred.pkl")
    res.update(model=name, seed=a.seed, config=str(CONFIGS[base]), model_cfg=model_cfg, arm=arm, yearly=info,
               check_2017_max_abs_diff=check, fit_sec=round(sum(v["fit_sec"] for v in info.values()), 1),
               total_sec=round(time.time() - t1, 1))
    tmp = out_dir / f"seed_{tag}.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    tmp.replace(out_dir / f"seed_{tag}.json")
    for k in ("label.pkl", "info.json"):
        stash[k].unlink(missing_ok=True)
    print(f"[{name} seed {tag}] IC={res['IC']:.4f} RankIC={res['Rank IC']:.4f} exc_w_cost={res['ann_excess_w_cost']:.4f} "
          f"IR={res['ir_w_cost']:.3f} check2017={check} sec={res['total_sec']}", flush=True)


if __name__ == "__main__":
    main()
