# -*- coding: utf-8 -*-
"""T21: purged-CV choice of the epoch count under three validation criteria (MSE / IC / Rank IC), Qlib LSTM / ALSTM.

Fold records: ~/ext/runs/<m>_cvcrit/seed_XX_fK.json (mse_curve = Qlib -MSE, ic_curve, ric_curve; 20 epochs each).
Arms (~/ext/runs/<name>/seed_XX.json):
  es / fb5 / fb10   official early stopping, bare fixed budget 5 / 10 epochs
  cv                e* by 5-fold mean -MSE, retrained on 2008-2016 (T19: <m>_cv)
  cvric / cvic      e* by 5-fold mean Rank IC / IC, retrained on 2008-2016 (T21: <m>_cvric, <m>_cvic)

Writes results/external/seeds_cv_criteria.md / .csv and results/external/cv_criteria.md.

    .venv/bin/python external/cv_criteria.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from cv_vs_es import MODELS, OUT, N_FOLDS
from protocol_vs_es import OVERALL, RUNS, YEARLY, YEARS, load, pair_rows

ARMS = {"es": "{m}", "fb5": "{m}_fb5", "fb10": "{m}_fb10", "cv": "{m}_cv", "cvric": "{m}_cvric", "cvic": "{m}_cvic"}
ARM_NAMES = {"es": "官方早停", "fb5": "裸固定 5 轮", "fb10": "裸固定 10 轮", "cv": "CV-MSE 定轮",
             "cvric": "CV-RankIC 定轮", "cvic": "CV-IC 定轮"}
CRITS = {"mse": ("mse_curve", "−MSE"), "ic": ("ic_curve", "IC"), "ric": ("ric_curve", "Rank IC")}
PAIRS = [("cvric", "cv"), ("cvric", "es"), ("cvric", "fb5")]


def crit_folds(m: str) -> dict:
    out = {}
    for f in sorted((RUNS / f"{m}_cvcrit").glob("seed_[0-9][0-9]_f[0-9].json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.setdefault(r["seed"], {})[r["fold"]] = r
    return {s: v for s, v in out.items() if len(v) == N_FOLDS}


def e_stars(fd: dict) -> dict:
    """seed -> {crit: (e*, fold-best list)}; e* is 1-based argmax of the 5-fold mean curve."""
    out = {}
    for s, v in fd.items():
        out[s] = {}
        for c, (key, _) in CRITS.items():
            curves = np.array([v[k][key] for k in range(N_FOLDS)])
            out[s][c] = (int(curves.mean(axis=0).argmax()) + 1, [int(x.argmax()) + 1 for x in curves])
    return out


def seeds_table(es_all: dict, data_all: dict) -> None:
    cols = ["model", "seed", "e_mse", "e_ic", "e_ric", "fold_best_mse", "fold_best_ic", "fold_best_ric"]
    for arm in ("cv", "cvric", "cvic"):
        cols += [f"{arm}_{k}" for k in ("IC", "Rank IC", "ann_excess_w_cost", "ir_w_cost")]
        cols += [f"{arm}_ann_excess_w_cost_{y}" for y in YEARS]
    rows = []
    for m in MODELS:
        for s, e in sorted(es_all[m].items()):
            row = {"model": m, "seed": s}
            for c in CRITS:
                row[f"e_{c}"] = e[c][0]
                row[f"fold_best_{c}"] = "/".join(map(str, e[c][1]))
            for arm in ("cv", "cvric", "cvic"):
                r = data_all[m][arm].get(s)
                for k in ("IC", "Rank IC", "ann_excess_w_cost", "ir_w_cost"):
                    row[f"{arm}_{k}"] = r[k] if r else ""
                for y in YEARS:
                    row[f"{arm}_ann_excess_w_cost_{y}"] = r["years"][y]["ann_excess_w_cost"] if r else ""
            rows.append(row)
    with open(OUT / "seeds_cv_criteria.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    f4 = lambda v: f"{v:+.4f}" if v != "" else "—"
    L = ["# T21：三种验证准则下的 CV 定轮，逐种子结果", "",
         "e*_M / e*_I / e*_R 分别为按 5 折平均 −MSE / IC / Rank IC 最优的轮数；括号内为各折各自的最优轮。"
         "后三组为用 2008—2016 重训对应 e* 轮后的测试期结果（2017-01—2020-07，含成本）。完整数据见 seeds_cv_criteria.csv。", "",
         "| 模型 | 种子 | e*_M（各折） | e*_I（各折） | e*_R（各折） | CV-MSE IC | CV-IC IC | CV-RankIC IC | CV-MSE 超额 | CV-IC 超额 | CV-RankIC 超额 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {MODELS[r['model']]} | {r['seed']} | " + " | ".join(f"{r[f'e_{c}']}（{r[f'fold_best_{c}']}）" for c in ("mse", "ic", "ric"))
                 + " | " + " | ".join(f4(r[f"{a}_IC"]) for a in ("cv", "cvic", "cvric"))
                 + " | " + " | ".join(f4(r[f"{a}_ann_excess_w_cost"]) for a in ("cv", "cvic", "cvric")) + " |")
    (OUT / "seeds_cv_criteria.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def ms(v) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():+.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"


def main() -> None:
    es_all, data_all, fd_all = {}, {}, {}
    for m in MODELS:
        fd_all[m] = crit_folds(m)
        es_all[m] = e_stars(fd_all[m])
        data_all[m] = {k: load(v.format(m=m)) for k, v in ARMS.items()}
    seeds_table(es_all, data_all)
    L = ["# 论文六：换验证准则的 CV 定轮（T21）", "",
         "与 T19 同一套 5 折净化 CV（2008—2016，嵌入期 20 个交易日，每折 20 轮不早停），每轮同时记录验证期 −MSE、IC、Rank IC；"
         "e* 为 5 折平均曲线最优的轮数，选轮全程不看测试期。CV-RankIC / CV-IC 组用 2008—2016 重训对应 e* 轮（e* 与 T19 相同时复用 T19 的结果）。"
         "实现见 `external/run_cv.py`。表中“差”为前者减后者，p 为配对 t 检验，“占优”为前者更大的种子数，sd 为跨种子标准差。", ""]
    for m, mname in MODELS.items():
        fd, es, data = fd_all[m], es_all[m], data_all[m]
        diffs = [r.get("max_abs_diff_vs_t19_mse") for v in fd.values() for r in v.values()
                 if r.get("max_abs_diff_vs_t19_mse") is not None]
        L += [f"## {mname}", "", "### e* 的分布", ""]
        if 0 in fd and 0 in fd[0]:
            L.append(f"- 复现核对：种子 0 折 0 的 −MSE 曲线与 T19 的最大绝对差 {fd[0][0].get('max_abs_diff_vs_t19_mse')}"
                     + (f"（全部 {len(diffs)} 个有 T19 对应的折中最大 {max(diffs):.2e}）" if diffs else "") + "。")
        mism = [s for s, e in es.items() if s in data["cvric"] and data["cvric"][s]["e_star"] != e["ric"][0]]
        mism += [s for s, e in es.items() if s in data["cvic"] and data["cvic"][s]["e_star"] != e["ic"][0]]
        mism += [s for s, e in es.items() if s in data["cv"] and data["cv"][s]["e_star"] != e["mse"][0]]
        L.append(f"- 折记录算出的 e* 与重训组记录的 e* 不一致的种子：{sorted(set(mism)) or '无'}。")
        for c, (_, cname) in CRITS.items():
            e = np.array([v[c][0] for v in es.values()])
            fb = np.array([b for v in es.values() for b in v[c][1]])
            L.append(f"- 准则 {cname}：e*（{len(e)} 个种子）均值 {e.mean():.1f}，中位数 {np.median(e):g}，范围 {e.min()}—{e.max()}，"
                     f"落在 5—10 轮 {int(((e >= 5) & (e <= 10)).sum())}/{len(e)}；逐种子 " + "，".join(f"{s}:{v[c][0]}" for s, v in sorted(es.items()))
                     + f"；各折最优轮（{len(fb)} 折）均值 {fb.mean():.1f}，" + "，".join(f"第 {k} 轮 {int((fb == k).sum())} 次" for k in sorted(set(fb.tolist()))))
        L.append("")
        any_v = next(iter(fd.values()))
        for c, (key, cname) in CRITS.items():
            curves = np.array([np.mean([v[k][key] for k in range(N_FOLDS)], axis=0) for v in fd.values()])
            per_fold = {k: np.array([v[k][key] for v in fd.values()]).mean(axis=0) for k in range(N_FOLDS)}
            L += [f"### 5 折验证曲线：{cname}（{len(fd)} 个种子，越大越好）", "",
                  "| 轮 | 5 折平均（跨种子均值 ± sd） | " + " | ".join(
                      f"折 {k}（{any_v[k]['valid_start'][:7]}—{any_v[k]['valid_end'][:7]}）" for k in range(N_FOLDS)) + " |",
                  "|---|---|" + "---|" * N_FOLDS]
            for i in range(curves.shape[1]):
                L.append(f"| {i + 1} | {curves[:, i].mean():.5f} ± {curves[:, i].std(ddof=1) if len(curves) > 1 else 0:.5f} | "
                         + " | ".join(f"{per_fold[k][i]:.5f}" for k in range(N_FOLDS)) + " |")
            L.append("")
        seeds = sorted(es)
        L += ["### 各组整体（均值 ± 跨种子 sd）", "", "| 组 | 种子数 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|"]
        for k in ARMS:
            d = [data[k][s] for s in seeds if s in data[k]]
            if d:
                L.append(f"| {ARM_NAMES[k]} | {len(d)} | " + " | ".join(ms([r[key] for r in d]) for key, _ in OVERALL) + " |")
        L += ["", "### 各组分年（IC / 年化超额含成本，均值）", "", "| 组 | " + " | ".join(f"{y} IC | {y} 超额" for y in YEARS) + " |",
              "|---|" + "---|---|" * len(YEARS)]
        for k in ARMS:
            d = [data[k][s] for s in seeds if s in data[k]]
            if d:
                L.append(f"| {ARM_NAMES[k]} | " + " | ".join(
                    f"{np.mean([r['years'][y]['IC'] for r in d]):+.4f} | {np.mean([r['years'][y]['ann_excess_w_cost'] for r in d]):+.4f}"
                    for y in YEARS) + " |")
        L.append("")
        for x, z in PAIRS:
            dx = {s: data[x][s] for s in seeds if s in data[x]}
            dz = {s: data[z][s] for s in seeds if s in data[z]}
            head = f"| 指标 | {ARM_NAMES[z]} | {ARM_NAMES[x]} | 差 | p | 占优 | sd（{ARM_NAMES[z]}） | sd（{ARM_NAMES[x]}） |"
            L += [f"### {ARM_NAMES[x]} 对 {ARM_NAMES[z]}（{len(set(dx) & set(dz))} 对种子）", "", head, "|---|---|---|---|---|---|---|---|"]
            L += pair_rows(dx, dz, OVERALL)
            for y in YEARS:
                L += ["", f"**{y} 年**", "", head, "|---|---|---|---|---|---|---|---|"]
                L += pair_rows(dx, dz, YEARLY, y)
            L.append("")
    (OUT / "cv_criteria.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(l for l in L if l.startswith(("##", "- ", "| CV", "| 官方", "| 裸"))))


if __name__ == "__main__":
    main()
