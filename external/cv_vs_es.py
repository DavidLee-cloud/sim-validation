# -*- coding: utf-8 -*-
"""T19: purged-CV choice of the epoch count vs official early stopping on Qlib LSTM / ALSTM (paired by seed).

Arms (~/ext/runs/<name>/seed_XX.json):
  es        official early stopping (T14: lstm, alstm)
  fb5/fb10  bare fixed budget 5 / 10 epochs (T18-A: <m>_fb5, <m>_fb10)
  cv        e* from 5-fold purged CV on 2008-2016, retrained on 2008-2016 for e* epochs (T19: <m>_cv)
  cvctrl    control: e* epochs on the official training segment 2008-2014 only (T19: <m>_cvctrl)
Fold records: ~/ext/runs/<m>_cvfold/seed_XX_fK.json.

Writes results/external/seeds_cv_epochs.md / .csv (per seed: e*, best epoch of each fold, overall and per-year metrics
of cv and cvctrl) and results/external/cv_vs_es.md (e* distribution, 5-fold mean validation curves, paired comparisons
cv vs es / fb5 / fb10 / cvctrl and cvctrl vs es, overall and per year).

    .venv/bin/python external/cv_vs_es.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from protocol_vs_es import OVERALL, RUNS, YEARLY, YEARS, load, pair_rows

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "external"
MODELS = {"lstm": "LSTM", "alstm": "ALSTM"}
ARMS = {"es": "{m}", "fb5": "{m}_fb5", "fb10": "{m}_fb10", "cv": "{m}_cv", "cvctrl": "{m}_cvctrl"}
ARM_NAMES = {"es": "官方早停", "fb5": "裸固定 5 轮", "fb10": "裸固定 10 轮", "cv": "CV 定轮（2008—2016 重训）",
             "cvctrl": "CV 定轮对照（仅 2008—2014）"}
PAIRS = [("cv", "es"), ("cv", "fb5"), ("cv", "fb10"), ("cv", "cvctrl"), ("cvctrl", "es")]
N_FOLDS, E_MAX = 5, 20


def folds(m: str) -> dict:
    out = {}
    for f in sorted((RUNS / f"{m}_cvfold").glob("seed_[0-9][0-9]_f[0-9].json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.setdefault(r["seed"], {})[r["fold"]] = r
    return out


def seeds_table() -> None:
    cols = ["model", "arm", "seed", "e_star", "fold_best", "IC", "ICIR", "Rank IC", "Rank ICIR", "ann_excess_w_cost",
            "ir_w_cost", "mdd_w_cost"] + [f"{k}_{y}" for y in YEARS for k in ("IC", "Rank IC", "ann_excess_w_cost")]
    rows = []
    for m in MODELS:
        for arm in ("cv", "cvctrl"):
            for s, r in sorted(load(ARMS[arm].format(m=m)).items()):
                row = {"model": m, "arm": arm, "seed": s, "e_star": r["e_star"], "fold_best": "/".join(map(str, r["fold_best"]))}
                row.update({k: r[k] for k in cols[5:12]})
                for y in YEARS:
                    for k in ("IC", "Rank IC", "ann_excess_w_cost"):
                        row[f"{k}_{y}"] = r["years"][y][k]
                rows.append(row)
    with open(OUT / "seeds_cv_epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    L = ["# T19：净化交叉验证定训练轮数，逐种子结果", "",
         "e* 为 5 折平均验证分数最优的轮数；“各折最优”为各折各自的最优轮。cv＝用 2008—2016 全部数据重训 e* 轮；"
         "cvctrl＝只用官方训练段 2008—2014 训练 e* 轮（对照）。指标同前（测试期 2017-01—2020-07，含成本）。完整数据见 seeds_cv_epochs.csv。", "",
         "| 模型 | 组 | 种子 | e* | 各折最优 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） | " + " | ".join(f"{y} 年化超额" for y in YEARS) + " |",
         "|---|---|---|---|---|---|---|---|---|" + "---|" * len(YEARS)]
    for r in rows:
        L.append(f"| {MODELS[r['model']]} | {r['arm']} | {r['seed']} | {r['e_star']} | {r['fold_best']} | {r['IC']:+.4f} | "
                 f"{r['Rank IC']:+.4f} | {r['ann_excess_w_cost']:+.4f} | {r['ir_w_cost']:+.3f} | "
                 + " | ".join(f"{r[f'ann_excess_w_cost_{y}']:+.4f}" for y in YEARS) + " |")
    (OUT / "seeds_cv_epochs.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    seeds_table()
    L = ["# 论文六：净化交叉验证定训练轮数（T19）", "",
         "模型 Qlib LSTM、ALSTM（官方 Alpha158 配置，恒定学习率），各组同种子配对；测试期 2017-01 至 2020-07，回测与成本同官方。"
         "2008-01—2016-12 按交易日切成 5 个相邻折，验证折两侧各 20 个交易日的样本从训练集剔除；每折新初始化、训练 20 轮不早停，"
         "记录每轮验证分数（Qlib 的 −MSE）；e* 为 5 折平均分数最优的轮数。选 e* 全程不看测试期。实现见 `external/run_cv.py`。",
         "表中“差”为前者减后者，p 为配对 t 检验，“占优”为前者更大的种子数，sd 为跨种子标准差。", ""]
    for m, mname in MODELS.items():
        data = {k: load(v.format(m=m)) for k, v in ARMS.items()}
        fd = folds(m)
        L += [f"## {mname}", ""]
        cv = data["cv"]
        if cv:
            e = np.array([r["e_star"] for r in cv.values()])
            fb = np.array([b for r in cv.values() for b in r["fold_best"]])
            L += ["### e* 的分布", "",
                  f"- e*（{len(e)} 个种子）：均值 {e.mean():.1f}，中位数 {np.median(e):.0f}，范围 {e.min()}—{e.max()}，跨种子 sd {e.std(ddof=1) if len(e) > 1 else 0:.2f}；"
                  f"落在 5—10 轮的种子 {int(((e >= 5) & (e <= 10)).sum())}/{len(e)}。",
                  "- 逐种子 e*：" + "，".join(f"{s}:{r['e_star']}" for s, r in sorted(cv.items())),
                  f"- 各折各自的最优轮（{len(fb)} 个折）：均值 {fb.mean():.1f}，范围 {fb.min()}—{fb.max()}；"
                  + "，".join(f"第 {k} 轮 {int((fb == k).sum())} 次" for k in sorted(set(fb.tolist()))), ""]
        if fd:
            done = {s: v for s, v in fd.items() if len(v) == N_FOLDS}
            if done:
                curves = np.array([np.mean([v[k]["valid_curve"] for k in range(N_FOLDS)], axis=0) for v in done.values()])
                per_fold = {k: np.array([v[k]["valid_curve"] for v in done.values()]).mean(axis=0) for k in range(N_FOLDS)}
                head = "| 轮 | 5 折平均（跨种子均值 ± sd） | " + " | ".join(
                    f"折 {k}（{next(iter(done.values()))[k]['valid_start'][:7]}—{next(iter(done.values()))[k]['valid_end'][:7]}）" for k in range(N_FOLDS)) + " |"
                L += [f"### 5 折验证曲线（{len(done)} 个种子；验证分数＝−MSE，越大越好）", "", head, "|---|---|" + "---|" * N_FOLDS]
                for i in range(curves.shape[1]):
                    L.append(f"| {i + 1} | {curves[:, i].mean():.5f} ± {curves[:, i].std(ddof=1) if len(curves) > 1 else 0:.5f} | "
                             + " | ".join(f"{per_fold[k][i]:.5f}" for k in range(N_FOLDS)) + " |")
                L.append("")
        L += ["### 各组整体（均值 ± 跨种子 sd）", "", "| 组 | 种子数 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|"]
        seeds = sorted(cv)
        for k in ARMS:
            d = {s: data[k][s] for s in seeds if s in data[k]}
            if d:
                L.append(f"| {ARM_NAMES[k]} | {len(d)} | " + " | ".join(
                    f"{np.mean([r[key] for r in d.values()]):+.4f} ± {np.std([r[key] for r in d.values()], ddof=1) if len(d) > 1 else 0:.4f}"
                    for key, _ in OVERALL) + " |")
        L.append("")
        for x, z in PAIRS:
            if not data[x] or not data[z]:
                L += [f"### {ARM_NAMES[x]} 对 {ARM_NAMES[z]}：尚无结果", ""]
                continue
            head = f"| 指标 | {ARM_NAMES[z]} | {ARM_NAMES[x]} | 差 | p | 占优 | sd（{ARM_NAMES[z]}） | sd（{ARM_NAMES[x]}） |"
            L += [f"### {ARM_NAMES[x]} 对 {ARM_NAMES[z]}（{len(set(data[x]) & set(data[z]))} 对种子）", "", head, "|---|---|---|---|---|---|---|---|"]
            L += pair_rows(data[x], data[z], OVERALL)
            for y in YEARS:
                L += ["", f"**{y} 年**", "", head, "|---|---|---|---|---|---|---|---|"]
                L += pair_rows(data[x], data[z], YEARLY, y)
            L.append("")
    (OUT / "cv_vs_es.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
