# -*- coding: utf-8 -*-
"""P7 外部复现：挑选层与区间层（2026-10-03，本机运行，只读已提交的逐种子逐年指标）。

口径（计算前写定）：
- 指标：年化超额（含成本）；开发段＝2017—2018，样本外段＝2019—2020-07，均按交易日加权合成。
- 挑选层：研究者在开发段为每个模型跑 s 个种子（s＝1、3、5），挑开发均值最高者。随机抽 5000 次，报告：
  赢家的开发估计；同一开发段、其余种子上的均值（种子层高估）；样本外 20 种子均值（区间层再缩水）；
  赢家是否为样本外 20 种子均值最高的模型。
- 区间层：5 个模型按 20 种子均值逐年排名，报告逐年最优模型与年份间排名的 Kendall τ。
- 单种子反转：任取两模型各一个种子，其开发段次序与 20 种子均值次序相反的概率。
输出：results/external/p7_layers.md
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "external"
df = pd.concat([pd.read_csv(D / "seeds_lgb_mlp.csv"), pd.read_csv(D / "seeds_gru_lstm_alstm.csv")], ignore_index=True)


def seg(r, years):
    w = np.array([r[f"{y}_n_days"] for y in years], float)
    x = np.array([r[f"{y}_ann_excess_w_cost"] for y in years], float)
    return float((w * x).sum() / w.sum())


df["dev"] = df.apply(lambda r: seg(r, (2017, 2018)), axis=1)
df["oos"] = df.apply(lambda r: seg(r, (2019, 2020)), axis=1)
models = sorted(df.model.unique())
seeds = {m: df[df.model == m].set_index("seed") for m in models}
full_dev = {m: seeds[m].dev.mean() for m in models}
full_oos = {m: seeds[m].oos.mean() for m in models}
best_oos = max(full_oos, key=full_oos.get)
rng = np.random.default_rng(0)
out = ["# P7 外部复现：挑选层与区间层（Qlib 沪深300 Alpha158，5 模型 × 20 种子）", "",
       "年化超额（含成本）。开发段 2017—2018，样本外段 2019—2020-07。脚本 `external/p7_layers.py`。", "",
       "## 20 种子均值", "", "| 模型 | 开发段 | 样本外段 | 开发段种子 sd |", "|---|---|---|---|"]
for m in models:
    out.append(f"| {m} | {100 * full_dev[m]:.2f}% | {100 * full_oos[m]:.2f}% | {100 * seeds[m].dev.std(ddof=1):.2f} |")
out += ["", "## 挑选层（随机抽 5000 次）", "",
        "| 每模型种子数 s | 赢家开发估计 | 赢家同段其余种子均值 | 种子层高估 | 赢家样本外 20 种子均值 | 样本外最优模型均值 | 选中样本外最优的概率 | 选中各模型的频率 |",
        "|---|---|---|---|---|---|---|---|"]
for s in (1, 3, 5):
    dv, rest, oo, hit, freq = [], [], [], 0, {m: 0 for m in models}
    for _ in range(5000):
        pick = {m: rng.choice(seeds[m].index, s, replace=False) for m in models}
        est = {m: seeds[m].loc[pick[m], "dev"].mean() for m in models}
        w = max(est, key=est.get)
        freq[w] += 1
        dv.append(est[w]); rest.append(seeds[w].drop(pick[w]).dev.mean()); oo.append(full_oos[w]); hit += w == best_oos
    f = "，".join(f"{m} {100 * freq[m] / 5000:.0f}%" for m in models)
    out.append(f"| {s} | {100 * np.mean(dv):.2f}% | {100 * np.mean(rest):.2f}% | {100 * (np.mean(dv) - np.mean(rest)):+.2f} pp | "
               f"{100 * np.mean(oo):.2f}% | {100 * full_oos[best_oos]:.2f}%（{best_oos}） | {100 * hit / 5000:.0f}% | {f} |")
yrs = [2017, 2018, 2019, 2020]
rk = pd.DataFrame({y: {m: seeds[m][f"{y}_ann_excess_w_cost"].mean() for m in models} for y in yrs})
out += ["", "## 区间层：逐年 20 种子均值与排名", "", "| 模型 | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
for m in models:
    out.append(f"| {m} | " + " | ".join(f"{100 * rk.loc[m, y]:.2f}%（第{int(rk[y].rank(ascending=False)[m])}）" for y in yrs) + " |")
taus = [kendalltau(rk[a], rk[b]).correlation for a, b in itertools.combinations(yrs, 2)]
out += ["", f"逐年最优：" + "，".join(f"{y} {rk[y].idxmax()}" for y in yrs) + f"；年份间排名 Kendall τ 均值 {np.mean(taus):.2f}（范围 {min(taus):.2f}—{max(taus):.2f}）。"]
rev, tot = 0, 0
for a, b in itertools.combinations(models, 2):
    sign = np.sign(full_dev[a] - full_dev[b])
    for x in seeds[a].dev:
        for y in seeds[b].dev:
            tot += 1; rev += np.sign(x - y) != sign
out += ["", f"## 单种子反转", "", f"任取两模型各一个种子，开发段次序与 20 种子均值次序相反的概率：{100 * rev / tot:.0f}%（{tot} 对）。"]
(D / "p7_layers.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
