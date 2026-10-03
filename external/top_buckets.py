# -*- coding: utf-8 -*-
"""T16: does the top of the score ranking underperform? (public Qlib models, read-only)

For each model x seed, the test-period daily scores (~/ext/runs/<model>/seed_XX_pred.pkl, from run_seeds.py)
are ranked each day in descending order among the CSI300 members that have a score that day (the pred index
is already the CSI300 universe of that day). Each rank bucket's mean label minus that day's mean label over all
ranked stocks with a label gives the bucket's daily excess. Labels come from the qlib data interface:
  1-day  = Ref($close,-2)/Ref($close,-1)-1   (the Alpha158 label)
  20-day = Ref($close,-21)/Ref($close,-1)-1  (bought at the next close, held 20 trading days)
Annualized: 1-day x252, 20-day x252/20.

Check: the daily Pearson IC of score vs the 1-day label is recomputed and compared with the IC in seed_XX.json.
Also records the MLP training-segment row count (for T15's epoch equivalent) in results/external/label_meta.json.

    ~/ext/.venv-qlib/bin/python external/top_buckets.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "ext" / "runs"
OUT = ROOT / "results" / "external"
MODELS = ["lgb", "mlp", "gru", "lstm", "alstm"]
NAMES = {"lgb": "LightGBM", "mlp": "MLP", "gru": "GRU", "lstm": "LSTM", "alstm": "ALSTM"}
BUCKETS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), ("51-100", 51, 100), ("101+", 101, 10**9)]
LABELS = {"1d": ("Ref($close, -2)/Ref($close, -1) - 1", 252.0), "20d": ("Ref($close, -21)/Ref($close, -1) - 1", 252.0 / 20)}
YEARS = (2017, 2018, 2019, 2020)


def load_labels() -> pd.DataFrame:
    from qlib.data import D
    df = D.features(D.instruments("csi300"), [LABELS[k][0] for k in LABELS], start_time="2017-01-01", end_time="2020-08-01")
    df.columns = list(LABELS)
    return df.swaplevel().sort_index()                     # (datetime, instrument), like pred.pkl


def mlp_train_rows() -> int:
    """rows the MLP actually trains on (official config's handler, learn processors, train segment)."""
    from qlib.cli.run import render_template
    from qlib.utils import init_instance_by_config
    from ruamel.yaml import YAML
    from qlib.data.dataset.handler import DataHandlerLP
    path = Path.home() / "ext/qlib/examples/benchmarks/MLP/workflow_config_mlp_Alpha158.yaml"
    cfg = YAML(typ="safe", pure=True).load(render_template(str(path)))
    ds = init_instance_by_config(cfg["task"]["dataset"])
    return int(len(ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)))


def bucket_daily(score: pd.Series, label: pd.Series) -> pd.DataFrame:
    """daily excess mean label per rank bucket (columns = bucket names, index = datetime)."""
    d = pd.DataFrame({"s": score, "y": label}).dropna(subset=["s"])
    d["rank"] = d.groupby(level="datetime")["s"].rank(ascending=False, method="first")
    d["ex"] = d["y"] - d.groupby(level="datetime")["y"].transform("mean")
    out = {}
    for name, lo, hi in BUCKETS:
        sel = d[(d["rank"] >= lo) & (d["rank"] <= hi)]
        out[name] = sel.groupby(level="datetime")["ex"].mean()
    return pd.DataFrame(out)


def main() -> None:
    import qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    labels = load_labels()
    meta = {"label_1d": LABELS["1d"][0], "label_20d": LABELS["20d"][0], "label_rows": int(len(labels))}
    rows, checks = [], []
    for m in MODELS:
        for f in sorted((RUNS / m).glob("seed_[0-9]*_pred.pkl")):
            seed = int(f.name[5:7])
            pred = pd.read_pickle(f)["score"]
            lab = labels.reindex(pred.index)
            ic = pd.DataFrame({"s": pred, "y": lab["1d"]}).groupby(level="datetime").apply(lambda g: g["s"].corr(g["y"]))
            stored = json.loads((RUNS / m / f"seed_{seed:02d}.json").read_text())["IC"]
            checks.append(abs(ic.mean() - stored))
            for key, (_, ann) in LABELS.items():
                daily = bucket_daily(pred, lab[key])
                daily["top_minus_11_20"] = daily["1-5"] - daily["11-20"]
                for period in ["all", *YEARS]:
                    dd = daily if period == "all" else daily[daily.index.year == period]
                    r = {"model": m, "seed": seed, "label": key, "period": str(period), "n_days": len(dd)}
                    r.update({c: dd[c].mean() * ann for c in dd.columns})
                    rows.append(r)
            print(f"{m} seed {seed:02d}: IC recomputed {ic.mean():.6f} vs stored {stored:.6f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "top_buckets.csv", index=False, float_format="%.6f")
    meta["ic_check_max_abs_diff"] = float(max(checks))
    meta["n_train_rows"] = mlp_train_rows()
    (OUT / "label_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    write_md(df, meta)


def write_md(df: pd.DataFrame, meta: dict) -> None:
    cols = [b[0] for b in BUCKETS]
    L = ["# 预测极值论文：公开模型的分数顶端是否跑输（T16）", "",
         "数据：Qlib 官方配置 5 个模型 × 20 个种子的测试期（2017-01-03 至 2020-07-31）逐日预测分数（`external/run_seeds.py`）。",
         f"每个交易日在当日有分数的沪深300 成分股内按分数降序排名；各档的超额 = 档内平均标签 − 当日全体（有标签者）平均标签。",
         f"标签经 qlib 数据接口取得：1 日 `{meta['label_1d']}`（Alpha158 标签），20 日 `{meta['label_20d']}`。",
         "年化：1 日标签 ×252，20 日标签 ×252/20（20 日标签逐日重叠，均值按日平均后再折算）。",
         f"核对：按 1 日标签重算的逐日 IC 均值与 seed_XX.json 中 Qlib 报告的 IC 最大相差 {meta['ic_check_max_abs_diff']:.2e}。", ""]
    for key, title in (("1d", "1 日标签"), ("20d", "20 日累计标签")):
        d = df[df.label == key]
        L += [f"## {title}", "", "### ① 各档年化超额（20 个种子均值）", "",
              "| 模型 | " + " | ".join(f"第 {c} 名" for c in cols) + " |", "|---|" + "---|" * len(cols)]
        a = d[d.period == "all"]
        for m in MODELS:
            g = a[a.model == m]
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{g[c].mean():+.4f}" for c in cols) + " |")
        L += ["", "### ② 第 1—5 名 − 第 11—20 名（年化）", "",
              "| 模型 | 均值 | 跨种子 sd | 为负的种子数 |", "|---|---|---|---|"]
        for m in MODELS:
            v = a[a.model == m]["top_minus_11_20"]
            L.append(f"| {NAMES[m]} | {v.mean():+.4f} | {v.std(ddof=1):.4f} | {int((v < 0).sum())}/{len(v)} |")
        L += ["", "### ③ 按年份：第 1—5 名 − 第 11—20 名（均值 ± sd，为负的种子数）", "",
              "| 模型 | " + " | ".join(str(y) for y in YEARS) + " |", "|---|" + "---|" * len(YEARS)]
        for m in MODELS:
            cells = []
            for y in YEARS:
                v = d[(d.model == m) & (d.period == str(y))]["top_minus_11_20"]
                cells.append(f"{v.mean():+.4f} ± {v.std(ddof=1):.4f}（{int((v < 0).sum())}/{len(v)}）")
            L.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["逐种子、逐年份的全部数值见 `top_buckets.csv`（列：model, seed, label, period, n_days, 各档, top_minus_11_20）。", ""]
    (OUT / "top_buckets.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
