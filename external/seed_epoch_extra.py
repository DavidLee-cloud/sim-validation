# -*- coding: utf-8 -*-
"""两项补充分析（2026-10-03，本机，只读已提交数据）。

A（论文六）：LSTM、ALSTM、GRU、MLP 各种子的验证早停选中轮次与测试期表现（IC、Rank IC、年化超额含成本）的关系：
   Spearman 相关；选中第 1 轮与其后选中的种子的均值对比。选中轮次取自 results/external/epoch_stats.md 的逐种子表。
B（P7）：同一模型内“在开发段（2017—2018）挑最好的种子”对后续（2019—2020-07）的高估：
   各模型种子的开发段与样本外段相关（种子优劣是否持续）；全部 20 种子中开发段最好者、及随机 5 个中最好者，
   其开发段优势与样本外优势（相对 20 种子均值）。
输出：results/external/seed_epoch_extra.md
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "external"
df = pd.concat([pd.read_csv(D / "seeds_lgb_mlp.csv"), pd.read_csv(D / "seeds_gru_lstm_alstm.csv")], ignore_index=True)


def seg(r, years):
    w = np.array([r[f"{y}_n_days"] for y in years], float)
    return float((w * np.array([r[f"{y}_ann_excess_w_cost"] for y in years], float)).sum() / w.sum())


df["dev"] = df.apply(lambda r: seg(r, (2017, 2018)), axis=1)
df["oos"] = df.apply(lambda r: seg(r, (2019, 2020)), axis=1)

text = (D / "epoch_stats.md").read_text(encoding="utf-8")
sel = {}
for block in re.split(r"\n## ", text)[1:]:
    name = block.split("（")[0].strip().lower()
    for m in re.finditer(r"^\| (\d+) \| (\d+)", block, re.M):
        sel[(name, int(m.group(1)))] = int(m.group(2))

out = ["# 补充分析：早停轮次与测试表现；同模型内挑最好种子的高估（2026-10-03，`external/seed_epoch_extra.py`）", "",
       "## A 论文六：选中轮次与测试期表现（20 种子）", "",
       "| 模型 | 选中轮次范围 | 与年化超额的 Spearman ρ（p） | 与 IC 的 ρ（p） | 选第 1 轮的种子：年化超额 / IC | 其余种子：年化超额 / IC |",
       "|---|---|---|---|---|---|"]
for m in ("mlp", "gru", "lstm", "alstm"):
    g = df[df.model == m].copy()
    g["sel"] = [sel.get((m, int(s)), np.nan) for s in g.seed]
    g = g.dropna(subset=["sel"])
    r1 = spearmanr(g.sel, g.ann_excess_w_cost); r2 = spearmanr(g.sel, g.IC)
    a, b = g[g.sel == 1], g[g.sel > 1]
    fa = f"{100 * a.ann_excess_w_cost.mean():.2f}% / {a.IC.mean():.4f}（{len(a)} 个）" if len(a) else "—"
    fb = f"{100 * b.ann_excess_w_cost.mean():.2f}% / {b.IC.mean():.4f}（{len(b)} 个）"
    out.append(f"| {m} | {int(g.sel.min())}—{int(g.sel.max())} | {r1.correlation:+.2f}（{r1.pvalue:.2f}） | {r2.correlation:+.2f}（{r2.pvalue:.2f}） | {fa} | {fb} |")
out += ["", "## B P7：同一模型内挑开发段最好的种子", "",
        "| 模型 | 种子开发段—样本外相关（Pearson，p） | 20 选 1 最好种子：开发段优势 | 其样本外优势 | 随机 5 选 1（5000 次）：开发段优势 | 其样本外优势 |",
        "|---|---|---|---|---|---|"]
rng = np.random.default_rng(0)
for m in ("lgb", "mlp", "gru", "lstm", "alstm"):
    g = df[df.model == m].reset_index(drop=True)
    c = pearsonr(g.dev, g.oos)
    i = g.dev.idxmax()
    da, oa = g.dev[i] - g.dev.mean(), g.oos[i] - g.oos.mean()
    d5, o5 = [], []
    for _ in range(5000):
        idx = rng.choice(len(g), 5, replace=False)
        j = idx[np.argmax(g.dev.values[idx])]
        d5.append(g.dev[j] - g.dev.mean()); o5.append(g.oos[j] - g.oos.mean())
    out.append(f"| {m} | {c[0]:+.2f}（{c[1]:.2f}） | {100 * da:+.2f} pp | {100 * oa:+.2f} pp | {100 * np.mean(d5):+.2f} pp | {100 * np.mean(o5):+.2f} pp |")
out += ["", "读法：若种子优劣不持续（相关≈0），挑开发段最好的种子在样本外的优势应≈0——“最好的种子”只是噪声。"]
(D / "seed_epoch_extra.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
