# -*- coding: utf-8 -*-
"""T17: fixed budget (20 epochs, deploy the last) vs validation early stopping (official) for Qlib LSTM / ALSTM.

Pairs ~/ext/runs/<m>_fb/seed_XX.json with ~/ext/runs/<m>/seed_XX.json (same seed 0-19, same data and
config apart from n_epochs / early_stop / deployed epoch) and reports, per model, the mean paired difference
(fixed - early stop), the paired t-test p value, the number of seeds where fixed budget is better, and the
cross-seed sd of each arm; overall and per year (2017-2020).

    .venv/bin/python external/fixed_vs_es.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "ext" / "runs"
MODELS = {"lstm": "LSTM", "alstm": "ALSTM"}
OVERALL = [("IC", "IC"), ("Rank IC", "Rank IC"), ("ann_excess_w_cost", "年化超额（含成本）"), ("ir_w_cost", "信息比（含成本）")]
YEARLY = [("IC", "IC"), ("Rank IC", "Rank IC"), ("ann_excess_w_cost", "年化超额（含成本）")]
YEARS = ("2017", "2018", "2019", "2020")


def load(model: str) -> dict:
    return {int(f.name[5:7]): json.loads(f.read_text(encoding="utf-8")) for f in sorted((RUNS / model).glob("seed_[0-9][0-9].json"))}


def row(name: str, es: np.ndarray, fb: np.ndarray) -> str:
    d = fb - es
    p = stats.ttest_rel(fb, es).pvalue if len(d) > 1 else float("nan")
    return (f"| {name} | {es.mean():+.4f} | {fb.mean():+.4f} | {d.mean():+.4f} | {p:.3f} | "
            f"{int((d > 0).sum())}/{len(d)} | {es.std(ddof=1):.4f} | {fb.std(ddof=1):.4f} |")


def main() -> None:
    curves = json.loads((ROOT / "results/external/epoch_curves.json").read_text(encoding="utf-8"))
    head = "| 指标 | 早停（官方） | 固定预算 | 差（固定−早停） | 配对 t 检验 p | 固定预算占优的种子 | 早停 sd | 固定预算 sd |"
    L = ["# 论文六：公开模型的固定预算对照（T17）", "",
         "固定预算 = 官方 Alpha158 配置只改 n_epochs 20、early_stop 1000（不触发早停），并部署第 20 轮的参数"
         "（`external/fixed_budget.py`：每轮训练后保存参数快照，Qlib 的 fit 回载验证最优轮之后再载入最后一轮）。"
         "早停 = 官方配置原样（T14 之后的 20 种子运行）。同种子 0—19 配对；数据、特征、学习率、批量、损失、回测与成本都相同。",
         "复现核对：单线程与默认线程、固定预算与早停在共同的前几轮验证分数完全一致（如 LSTM 种子 0 前两轮 −0.996278、−0.994863），"
         "两组只在训练长度和部署轮次上不同。", ""]
    for m, name in MODELS.items():
        es, fb = load(m), load(m + "_fb")
        seeds = sorted(set(es) & set(fb))
        sel = [curves[m][str(s)]["best_round"] for s in seeds]
        L += [f"## {name}（{len(seeds)} 对种子）", "",
              f"早停组选中轮次：均值 {np.mean(sel):.2f}，范围 {min(sel)}—{max(sel)}（T15）；固定预算组一律部署第 20 轮。", "",
              "### 整体", "", head, "|---|---|---|---|---|---|---|---|"]
        for k, label in OVERALL:
            L.append(row(label, np.array([es[s][k] for s in seeds]), np.array([fb[s][k] for s in seeds])))
        L += ["", "### 按年份", ""]
        for y in YEARS:
            L += [f"**{y} 年**", "", head, "|---|---|---|---|---|---|---|---|"]
            for k, label in YEARLY:
                L.append(row(label, np.array([es[s]["years"][y][k] for s in seeds]),
                             np.array([fb[s]["years"][y][k] for s in seeds])))
            L.append("")
        L += ["### 每个种子（年化超额含成本）", "", "| 种子 | 早停选中轮次 | IC 早停 / 固定 | 年化超额 早停 / 固定 | 信息比 早停 / 固定 |",
              "|---|---|---|---|---|"]
        for s, b in zip(seeds, sel):
            L.append(f"| {s} | {b} | {es[s]['IC']:.4f} / {fb[s]['IC']:.4f} | {es[s]['ann_excess_w_cost']:+.4f} / "
                     f"{fb[s]['ann_excess_w_cost']:+.4f} | {es[s]['ir_w_cost']:+.3f} / {fb[s]['ir_w_cost']:+.3f} |")
        L.append("")
    out = ROOT / "results" / "external" / "fixed_vs_es.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
