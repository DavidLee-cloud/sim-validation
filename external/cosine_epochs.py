# -*- coding: utf-8 -*-
"""T22: sensitivity of the fixed budget to the epoch count under cosine annealing, Qlib LSTM / ALSTM (paired by seed).

Arms (~/ext/runs/<name>/seed_XX.json):
  es                   official early stopping (<m>)
  fb5 / fb10 / fb20    bare fixed budget 5 / 10 / 20 epochs (T18-A: <m>_fb5, <m>_fb10, <m>_fb)
  c5 / d1 / c20        cosine-annealed fixed budget 5 / 10 / 20 epochs (T22: <m>_c5, T20: <m>_d1, T22: <m>_c20)
Check: lstm_c10 seed 0 (E=10 through the T22 path) vs lstm_d1 seed 0 predictions.

Writes results/external/seeds_cosine_epochs.md / .csv and results/external/cosine_epochs.md.

    .venv/bin/python external/cosine_epochs.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from cv_vs_es import MODELS, OUT
from protocol_vs_es import OVERALL, RUNS, YEARLY, YEARS, load, pair_rows

ARMS = {"es": "{m}", "fb5": "{m}_fb5", "fb10": "{m}_fb10", "fb20": "{m}_fb",
        "c5": "{m}_c5", "d1": "{m}_d1", "c20": "{m}_c20"}
ARM_NAMES = {"es": "官方早停", "fb5": "裸固定 5 轮", "fb10": "裸固定 10 轮", "fb20": "裸固定 20 轮",
             "c5": "退火 5 轮（C5）", "d1": "退火 10 轮（D1）", "c20": "退火 20 轮（C20）"}
BARE, COS = ("fb5", "fb10", "fb20"), ("c5", "d1", "c20")
PAIRS = [("c5", "es"), ("c20", "es"), ("c5", "d1"), ("c20", "d1")]


def ms(v) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():+.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"


def check_diff() -> float | None:
    a, b = RUNS / "lstm_c10" / "seed_00_pred.pkl", RUNS / "lstm_d1" / "seed_00_pred.pkl"
    if not (a.exists() and b.exists()):
        return None
    x, y = pd.read_pickle(a), pd.read_pickle(b)
    x, y = (x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x), (y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y)
    x, y = x.align(y, join="outer")
    return float((x - y).abs().max())


def main() -> None:
    allrows = []
    diff = check_diff()
    L = ["# 论文六：余弦退火下固定预算对轮数的敏感性（T22）", "",
         "模型 Qlib LSTM、ALSTM（官方 Alpha158 配置），种子 0—19，各组同种子配对；测试期 2017-01 至 2020-07，回测与成本同官方。"
         "退火组与 T20 的 D1 完全相同（官方 Adam、余弦学习率 lr_e＝lr0·(0.01＋0.99·0.5·(1＋cos(π·e/(E−1))))、不加权、一次训练、部署最后一轮），"
         "只改总轮数 E＝5／10／20，各自按 E 从头训练。实现见 `external/fixed_budget.py`、`external/run_seeds.py`。"
         "表中“差”为前者减后者，p 为配对 t 检验，“占优”为前者更大的种子数，sd 为跨种子标准差。", "",
         f"- 核对：经 T22 路径以 E＝10 训练的 LSTM 种子 0（lstm_c10）与 T20 的 lstm_d1 预测的最大绝对差 = {diff if diff is not None else '缺文件'}。", ""]
    verdicts = []
    for m, mname in MODELS.items():
        data = {k: load(v.format(m=m)) for k, v in ARMS.items()}
        seeds = sorted(set.intersection(*(set(d) for d in data.values())))
        data = {k: {s: d[s] for s in seeds} for k, d in data.items()}
        for k, d in data.items():
            for s, r in d.items():
                row = {"model": m, "arm": k, "seed": s, **{key: r[key] for key, _ in OVERALL}}
                for y in YEARS:
                    for key, _ in YEARLY:
                        row[f"{key}_{y}"] = r["years"][y][key]
                allrows.append(row)
        L += [f"## {mname}（{len(seeds)} 个种子，取各组共有的种子）", "", "### 各组整体（均值 ± 跨种子 sd）", "",
              "| 组 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|"]
        for k in ARMS:
            L.append(f"| {ARM_NAMES[k]} | " + " | ".join(ms([r[key] for r in data[k].values()]) for key, _ in OVERALL) + " |")
        L += ["", "### 各组逐年（均值）", "", "| 组 | " + " | ".join(f"{y} IC | {y} Rank IC | {y} 超额" for y in YEARS) + " |",
              "|---|" + "---|---|---|" * len(YEARS)]
        for k in ARMS:
            L.append(f"| {ARM_NAMES[k]} | " + " | ".join(
                " | ".join(f"{np.mean([r['years'][y][key] for r in data[k].values()]):+.4f}" for key, _ in YEARLY) for y in YEARS) + " |")
        L += ["", "### 汇总①：三档轮数之间的最大差（各组均值的极差）", "", "| 指标 | 裸固定 5／10／20 | 退火 5／10／20 | 退火／裸固定 |", "|---|---|---|---|"]
        spread = {}
        for key, kname in OVERALL:
            sb = np.ptp([np.mean([r[key] for r in data[k].values()]) for k in BARE])
            sc = np.ptp([np.mean([r[key] for r in data[k].values()]) for k in COS])
            spread[key] = (sb, sc)
            L.append(f"| {kname} | {sb:.4f} | {sc:.4f} | {sc / sb if sb else float('nan'):.2f} |")
        es_ic = np.mean([r["IC"] for r in data["es"].values()])
        cos_ic = {k: np.mean([r["IC"] for r in data[k].values()]) for k in COS}
        ok = spread["IC"][1] <= 0.5 * spread["IC"][0] and all(v >= es_ic for v in cos_ic.values())
        verdicts.append(f"{mname}：退火 IC 极差 {spread['IC'][1]:.4f} vs 裸固定 {spread['IC'][0]:.4f}（比 {spread['IC'][1] / spread['IC'][0]:.2f}）；"
                        f"三档 IC " + "／".join(f"{cos_ic[k]:.4f}" for k in COS) + f" vs 早停 {es_ic:.4f} → "
                        + ("“退火降低了对轮数的敏感度”" if ok else "不成立"))
        L += ["", f"判读（事先写定）：{verdicts[-1]}", "", "### 汇总②：C5、C20 对官方早停、对 D1", ""]
        for x, z in PAIRS:
            head = f"| 指标 | {ARM_NAMES[z]} | {ARM_NAMES[x]} | 差 | p | 占优 | sd（{ARM_NAMES[z]}） | sd（{ARM_NAMES[x]}） |"
            L += [f"#### {ARM_NAMES[x]} 对 {ARM_NAMES[z]}", "", head, "|---|---|---|---|---|---|---|---|"]
            L += pair_rows(data[x], data[z], OVERALL)
            for y in YEARS:
                L += ["", f"**{y} 年**", "", head, "|---|---|---|---|---|---|---|---|"]
                L += pair_rows(data[x], data[z], YEARLY, y)
            L.append("")
    (OUT / "cosine_epochs.md").write_text("\n".join(L), encoding="utf-8")
    cols = list(allrows[0])
    with open(OUT / "seeds_cosine_epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    S = ["# T22：余弦退火固定预算的轮数敏感性，逐种子结果", "", "测试期 2017-01—2020-07，含成本。完整数据（含逐年）见 seeds_cosine_epochs.csv。", "",
         "| 模型 | 组 | 种子 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|---|"]
    for r in allrows:
        if r["arm"] in ("c5", "c20"):
            S.append(f"| {MODELS[r['model']]} | {ARM_NAMES[r['arm']]} | {r['seed']} | {r['IC']:+.4f} | {r['Rank IC']:+.4f} | "
                     f"{r['ann_excess_w_cost']:+.4f} | {r['ir_w_cost']:+.3f} |")
    (OUT / "seeds_cosine_epochs.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    print("\n".join(l for l in L if l.startswith(("##", "- ", "判读", "| 官方", "| 裸", "| 退火", "| IC", "| Rank", "| 年化", "| 信息"))
                    and "**" not in l)[:6000])


if __name__ == "__main__":
    main()
