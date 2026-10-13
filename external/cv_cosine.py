# -*- coding: utf-8 -*-
"""T23: CV (Rank IC) choice of the epoch count combined with cosine annealing, Qlib LSTM / ALSTM, seeds 0-9 (paired).

Arms (~/ext/runs/<name>/seed_XX.json):
  es / fb5 / c5     official early stopping, bare fixed 5 epochs, cosine fixed 5 epochs on 2008-2014 (T22: <m>_c5)
  cvric             T21: e*_R epochs, constant lr, retrained on 2008-2016 (<m>_cvric)
  g1 (cvr_cos)      T23: e*_R epochs, cosine, 2008-2016 (<m>_cvrcos)
  g2 (f5_cos_0816)  T23: 5 epochs, cosine, 2008-2016 (<m>_f5cos)
Check: lstm_cvrchk seed 0 (cosine off, E = e*_R) vs lstm_cvric seed 0 predictions.

Writes results/external/seeds_cv_cosine.md / .csv and results/external/cv_cosine.md.

    .venv/bin/python external/cv_cosine.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from scipy import stats

from cosine_epochs import ms
from cv_vs_es import MODELS, OUT
from protocol_vs_es import OVERALL, RUNS, YEARLY, YEARS, load, pair_rows

ARMS = {"es": "{m}", "fb5": "{m}_fb5", "c5": "{m}_c5", "cvric": "{m}_cvric", "g1": "{m}_cvrcos", "g2": "{m}_f5cos"}
ARM_NAMES = {"es": "官方早停", "fb5": "裸固定 5 轮", "c5": "退火 5 轮（C5，2008—2014）", "cvric": "CV-RankIC 定轮（T21）",
             "g1": "G1 CV-RankIC 定轮＋退火", "g2": "G2 退火 5 轮（2008—2016）"}
PAIRS = [("g1", "c5"), ("g1", "cvric"), ("g1", "g2"), ("g1", "es")]
SEEDS = range(10)


def main() -> None:
    a = pd.read_pickle(RUNS / "lstm_cvrchk" / "seed_00_pred.pkl")["score"]
    b = pd.read_pickle(RUNS / "lstm_cvric" / "seed_00_pred.pkl")["score"]
    diff = float((a - b.reindex(a.index)).abs().max())
    L = ["# 论文六：交叉验证定轮（秩信息系数）与余弦退火合用（T23）", "",
         "模型 Qlib LSTM、ALSTM（官方 Alpha158 配置），种子 0—9，各组同种子配对；测试期 2017-01 至 2020-07，回测与成本同官方。"
         "G1、G2 均在 2008—2016 新初始化训练一次、部署最后一轮，官方 Adam、不加权、只开余弦退火"
         "（lr_e＝lr0·(0.01＋0.99·0.5·(1＋cos(π·e/(E−1))))）；G1 的 E＝T21 按验证 Rank IC 选出的 e*_R，G2 的 E＝5。"
         "实现见 `external/run_cv.py`（job cvrcos／f5cos）。表中“差”为前者减后者，p 为配对 t 检验，“占优”为前者更大的种子数。", "",
         f"- 核对：LSTM 种子 0 关掉退火、E＝e*_R 重训（cvrchk）与 T21 的 CV-RankIC 重训组预测最大绝对差 = {diff}。", ""]
    rows, verdict = [], {}
    for m, mname in MODELS.items():
        data = {k: {s: r for s, r in load(v.format(m=m)).items() if s in SEEDS} for k, v in ARMS.items()}
        e = {s: r["e_star"] for s, r in data["g1"].items()}
        n1 = sum(v == 1 for v in e.values())
        L += [f"## {mname}", "", f"- G1 的 E（逐种子 e*_R）：" + "，".join(f"{s}:{v}" for s, v in sorted(e.items()))
              + (f"；其中 {n1} 个种子 E＝1，按恒定学习率训练。" if n1 else "；无 E＝1 的种子。"), "",
              "### 各组整体（均值 ± 跨种子 sd）", "", "| 组 | 种子数 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|"]
        for k in ARMS:
            L.append(f"| {ARM_NAMES[k]} | {len(data[k])} | " + " | ".join(ms([r[key] for r in data[k].values()]) for key, _ in OVERALL) + " |")
        L += ["", "### 各组逐年（均值）", "", "| 组 | " + " | ".join(f"{y} IC | {y} Rank IC | {y} 超额" for y in YEARS) + " |",
              "|---|" + "---|---|---|" * len(YEARS)]
        for k in ARMS:
            L.append(f"| {ARM_NAMES[k]} | " + " | ".join(
                " | ".join(f"{np.mean([r['years'][y][key] for r in data[k].values()]):+.4f}" for key, _ in YEARLY) for y in YEARS) + " |")
        L.append("")
        for x, z in PAIRS:
            head = f"| 指标 | {ARM_NAMES[z]} | {ARM_NAMES[x]} | 差 | p | 占优 | sd（{ARM_NAMES[z]}） | sd（{ARM_NAMES[x]}） |"
            L += [f"### {ARM_NAMES[x]} 对 {ARM_NAMES[z]}", "", head, "|---|---|---|---|---|---|---|---|"]
            L += pair_rows(data[x], data[z], OVERALL)
            for y in YEARS:
                L += ["", f"**{y} 年**", "", head, "|---|---|---|---|---|---|---|---|"]
                L += pair_rows(data[x], data[z], YEARLY, y)
            L.append("")
        s = sorted(data["g1"])
        dic = np.mean([data["g1"][i]["IC"] - data["c5"][i]["IC"] for i in s])
        dx = np.array([data["g1"][i]["ann_excess_w_cost"] - data["c5"][i]["ann_excess_w_cost"] for i in s])
        p = stats.ttest_rel([data["g1"][i]["ann_excess_w_cost"] for i in s], [data["c5"][i]["ann_excess_w_cost"] for i in s]).pvalue
        verdict[mname] = (dic >= -0.002 and (p >= 0.10 or dx.mean() > 0), dic, dx.mean(), p)
        for k in ARMS:
            for i, r in data[k].items():
                rows.append({"model": m, "arm": k, "seed": i, "E": e.get(i) if k == "g1" else "",
                             **{key: r[key] for key, _ in OVERALL},
                             **{f"{key}_{y}": r["years"][y][key] for y in YEARS for key, _ in YEARLY}})
    ok = sum(v[0] for v in verdict.values())
    L += ["## 判读（事先写定）", ""] + [f"- {k}：G1−C5 IC {v[1]:+.4f}，年化超额 {v[2]:+.4f}（p={v[3]:.3f}）→ {'满足' if v[0] else '不满足'}"
                                       for k, v in verdict.items()]
    L.append("- 结论：" + ("“不看测试期即可达到退火固定 5 轮的水平”" if ok == 2 else "“部分成立”" if ok == 1 else "不成立"))
    (OUT / "cv_cosine.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / "seeds_cv_cosine.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S = ["# T23：交叉验证定轮＋退火，逐种子结果", "", "测试期 2017-01—2020-07，含成本。完整数据（含逐年及对照组）见 seeds_cv_cosine.csv。", "",
         "| 模型 | 组 | 种子 | E | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|---|---|"]
    S += [f"| {MODELS[r['model']]} | {ARM_NAMES[r['arm']]} | {r['seed']} | {r['E'] if r['arm'] == 'g1' else 5} | {r['IC']:+.4f} | "
          f"{r['Rank IC']:+.4f} | {r['ann_excess_w_cost']:+.4f} | {r['ir_w_cost']:+.3f} |" for r in rows if r["arm"] in ("g1", "g2")]
    (OUT / "seeds_cv_cosine.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    print("\n".join(l for l in L if l.startswith(("##", "- ", "| 官方", "| 裸", "| 退火", "| CV", "| G", "| IC", "| 年化")) and "**" not in l))


if __name__ == "__main__":
    main()
