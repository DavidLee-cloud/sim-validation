# -*- coding: utf-8 -*-
"""T20: one-at-a-time decomposition of the full fixed-budget protocol on Qlib LSTM / ALSTM (paired by seed 0-19).

Arms (~/ext/runs/<name>/seed_XX.json):
  base  bare fixed budget, 10 epochs (T18-A: <m>_fb10; Qlib's fit, Adam, constant lr, one training, uniform sampling)
  d1    base + cosine learning rate (<m>_d1)              d2  base + AdamW weight decay 0.03 (<m>_d2)
  d3    base + recency-weighted sampling, half-life 504 (<m>_d3)
  d4    base + yearly expanding-window retraining (<m>_d4, run_yearly.py)
  b10   all four together (T18-B10: <m>_proto10)          es  official early stopping (<m>)

Writes results/external/protocol_decomp.md: per model a summary table, each Dk vs base (paired mean difference, t-test
p, seeds ahead, per year), and the sum of the four single effects against B10 - base (an interaction check).

    .venv/bin/python external/protocol_decomp.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from protocol_vs_es import OVERALL, YEARLY, YEARS, load, pair_rows

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"lstm": "LSTM", "alstm": "ALSTM"}
ARMS = {"base": "{m}_fb10", "d1": "{m}_d1", "d2": "{m}_d2", "d3": "{m}_d3", "d4": "{m}_d4", "b10": "{m}_proto10", "es": "{m}"}
ARM_NAMES = {"base": "底：裸固定 10 轮", "d1": "D1 余弦退火", "d2": "D2 权重衰减（AdamW 0.03）", "d3": "D3 近期加权",
             "d4": "D4 按年重训", "b10": "B10 完整协议（四项全加）", "es": "官方早停"}
DS = ("d1", "d2", "d3", "d4")


def main() -> None:
    L = ["# 论文六：完整协议的部件拆分（T20）", "",
         "模型 Qlib LSTM、ALSTM（官方 Alpha158 配置），种子 0—19，同种子配对；测试期 2017-01 至 2020-07，回测与成本同官方。"
         "底为 T18-A 的裸固定 10 轮；D1—D4 各在底上只加协议的一项，其余与底完全相同；B10 为四项全加（T18-B10）。"
         "实现见 `external/fixed_budget.py`（开关）、`external/run_seeds.py`（D1—D3）、`external/run_yearly.py`（D4）。",
         "表中“差”为前者减后者，p 为配对 t 检验，“占优”为前者更大的种子数，sd 为跨种子标准差。", ""]
    for m, mname in MODELS.items():
        data = {k: load(v.format(m=m)) for k, v in ARMS.items()}
        L += [f"## {mname}", "", "### 各组整体（均值 ± 跨种子 sd）", "",
              "| 组 | 种子数 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|"]
        for k in ARMS:
            d = data[k]
            if d:
                L.append(f"| {ARM_NAMES[k]} | {len(d)} | " + " | ".join(
                    f"{np.mean([r[key] for r in d.values()]):+.4f} ± {np.std([r[key] for r in d.values()], ddof=1) if len(d) > 1 else 0:.4f}"
                    for key, _ in OVERALL) + " |")
        L += ["", "按年份（年化超额含成本，均值）：", "", "| 组 | " + " | ".join(YEARS) + " |", "|---|" + "---|" * len(YEARS)]
        for k in ARMS:
            d = data[k]
            if d:
                L.append(f"| {ARM_NAMES[k]} | " + " | ".join(
                    f"{np.mean([r['years'][y]['ann_excess_w_cost'] for r in d.values()]):+.4f}" for y in YEARS) + " |")
        L.append("")
        base = data["base"]
        for x in DS + ("b10",):
            if not data[x]:
                L += [f"### {ARM_NAMES[x]} 对 底：尚无结果", ""]
                continue
            head = f"| 指标 | 底 | {ARM_NAMES[x]} | 差 | p | 占优 | sd（底） | sd（{ARM_NAMES[x]}） |"
            L += [f"### {ARM_NAMES[x]} 对 底（{len(set(data[x]) & set(base))} 对种子）", "", head, "|---|---|---|---|---|---|---|---|"]
            L += pair_rows(data[x], base, OVERALL)
            for y in YEARS:
                L += ["", f"**{y} 年**", "", head, "|---|---|---|---|---|---|---|---|"]
                L += pair_rows(data[x], base, YEARLY, y)
            L.append("")
        seeds = sorted(set(base).intersection(*(set(data[k]) for k in DS + ("b10",))))
        if seeds:
            L += [f"### 交互检查：四项单独效应之和 对 B10−底（{len(seeds)} 个种子都有全部组）", "",
                  "逐种子计算 Σ_k (Dk − 底) 与 (B10 − 底)，再取均值；“残差”＝(B10 − 底) − Σ，p 为残差的单样本 t 检验（检验是否存在交互）。", "",
                  "| 指标 | D1 | D2 | D3 | D4 | 四项之和 Σ | B10 − 底 | 残差（交互） | p |", "|---|---|---|---|---|---|---|---|---|"]
            for key, label in OVERALL:
                b = np.array([base[s][key] for s in seeds])
                eff = {k: np.array([data[k][s][key] for s in seeds]) - b for k in DS}
                tot = sum(eff.values())
                full = np.array([data["b10"][s][key] for s in seeds]) - b
                res = full - tot
                p = stats.ttest_1samp(res, 0).pvalue if len(res) > 1 else float("nan")
                L.append(f"| {label} | " + " | ".join(f"{eff[k].mean():+.4f}" for k in DS) +
                         f" | {tot.mean():+.4f} | {full.mean():+.4f} | {res.mean():+.4f} | {p:.3f} |")
            L.append("")
    out = ROOT / "results" / "external" / "protocol_decomp.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
