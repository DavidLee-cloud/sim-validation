# -*- coding: utf-8 -*-
"""T18: full fixed-budget protocol vs early stopping on Qlib LSTM / ALSTM (paired by seed 0-19).

Arms (all ~/ext/runs/<name>/seed_XX.json):
  es    official early stopping (T14 runs: lstm, alstm)
  fb5 / fb10 / fb20   bare fixed budget, 5 / 10 / 20 epochs, deploy the last (T18-A: <m>_fb5, <m>_fb10; T17: <m>_fb)
  b20   full protocol, 20 epochs, yearly expanding-window retraining (T18-B: <m>_proto; stopped after seeds 0-4,
        reported descriptively)
  b10   full protocol, 10 epochs (T18-B10: <m>_proto10; 10 rather than A's best 5 so as not to choose the epoch count
        on the test period, though 10 was informed by A's epoch curve)
  esyr  official early stopping, yearly expanding-window retraining (T18-C: <m>_esyr)

Writes results/external/protocol_vs_es.md: the epoch curve of the bare fixed budget (A), a descriptive summary of
B20 (seeds 0-4) next to the other arms on the same seeds, and the paired comparisons b10 vs es, b10 vs fb10,
b10 vs esyr, esyr vs es, overall and per year; arms whose runs are missing are skipped.

    .venv/bin/python external/protocol_vs_es.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "ext" / "runs"
MODELS = {"lstm": "LSTM", "alstm": "ALSTM"}
ARMS = {"es": "{m}", "fb5": "{m}_fb5", "fb10": "{m}_fb10", "fb20": "{m}_fb", "b20": "{m}_proto", "b10": "{m}_proto10",
        "esyr": "{m}_esyr"}
ARM_NAMES = {"es": "官方早停", "fb5": "裸固定 5 轮", "fb10": "裸固定 10 轮", "fb20": "裸固定 20 轮（T17）",
             "b20": "完整协议 20 轮（B20）", "b10": "完整协议 10 轮（B10）", "esyr": "早停＋按年重训（C）"}
PAIRS = [("b10", "es"), ("b10", "fb10"), ("b10", "esyr"), ("esyr", "es")]
OVERALL = [("IC", "IC"), ("Rank IC", "Rank IC"), ("ann_excess_w_cost", "年化超额（含成本）"), ("ir_w_cost", "信息比（含成本）")]
YEARLY = [("IC", "IC"), ("Rank IC", "Rank IC"), ("ann_excess_w_cost", "年化超额（含成本）")]
YEARS = ("2017", "2018", "2019", "2020")


def load(name: str) -> dict:
    d = RUNS / name
    return {int(f.name[5:7]): json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("seed_[0-9][0-9].json"))} if d.exists() else {}


def val(r: dict, k: str, y: str | None = None) -> float:
    return r["years"][y][k] if y else r[k]


def pair_rows(a: dict, b: dict, items, y=None) -> list:
    seeds = sorted(set(a) & set(b))
    out = []
    for k, label in items:
        x = np.array([val(a[s], k, y) for s in seeds])
        z = np.array([val(b[s], k, y) for s in seeds])
        d = x - z
        p = stats.ttest_rel(x, z).pvalue if len(d) > 1 else float("nan")
        out.append(f"| {label} | {z.mean():+.4f} | {x.mean():+.4f} | {d.mean():+.4f} | {p:.3f} | {int((d > 0).sum())}/{len(d)} | "
                   f"{z.std(ddof=1):.4f} | {x.std(ddof=1):.4f} |")
    return out


def main() -> None:
    L = ["# 论文六：公开模型上的完整固定预算协议对照（T18）", "",
         "模型 Qlib LSTM、ALSTM（官方 Alpha158 配置），种子 0—19，各组同种子配对；测试期 2017-01 至 2020-07，回测与成本同官方。",
         "各组定义见 `external/protocol_vs_es.py` 文首；实现见 `external/fixed_budget.py`、`external/run_seeds.py`、`external/run_yearly.py`。",
         "B10 的轮数（10）参考了 A 的轮数曲线；不取 A 的最优点 5 轮，以免用测试期结果挑参数。表中“差”为前者减后者，p 为配对 t 检验，“占优”为前者更大的种子数，sd 为跨种子标准差。", ""]
    for m, mname in MODELS.items():
        data = {k: load(v.format(m=m)) for k, v in ARMS.items()}
        L += [f"## {mname}", "", "### A：裸固定预算的轮数曲线（部署最后一轮；均值 ± 跨种子 sd）", "",
              "| 组 | 种子数 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|---|"]
        for k in ("es", "fb5", "fb10", "fb20"):
            d = data[k]
            if not d:
                continue
            cells = []
            for key, _ in OVERALL:
                v = np.array([r[key] for r in d.values()])
                cells.append(f"{v.mean():+.4f} ± {v.std(ddof=1):.4f}")
            L.append(f"| {ARM_NAMES[k]} | {len(d)} | " + " | ".join(cells) + " |")
        L += ["", "按年份（年化超额含成本，均值）：", "", "| 组 | " + " | ".join(YEARS) + " |", "|---|" + "---|" * len(YEARS)]
        for k in ("es", "fb5", "fb10", "fb20"):
            d = data[k]
            if d:
                L.append(f"| {ARM_NAMES[k]} | " + " | ".join(f"{np.mean([r['years'][y]['ann_excess_w_cost'] for r in d.values()]):+.4f}"
                                                            for y in YEARS) + " |")
        L.append("")
        if data["b20"]:
            seeds = sorted(data["b20"])
            L += [f"### B20：完整协议 20 轮（种子 {seeds[0]}—{seeds[-1]}，共 {len(seeds)} 个；描述性，不做检验）", "",
                  "B20 只跑了种子 0—4 即停：A 的轮数曲线显示裸固定预算 20 轮在 Qlib 中训练过量，B20 的前 3 个种子与之一致，故改跑 B10。"
                  "下表各组都只取同一批种子（均值 ± sd）。", "",
                  "| 组 | IC | Rank IC | 年化超额（含成本） | 信息比（含成本） |", "|---|---|---|---|---|"]
            for k in ("es", "fb5", "fb10", "fb20", "b20", "b10", "esyr"):
                d = {s: data[k][s] for s in seeds if s in data[k]}
                if len(d) == len(seeds):
                    L.append(f"| {ARM_NAMES[k]} | " + " | ".join(
                        f"{np.mean([r[key] for r in d.values()]):+.4f} ± {np.std([r[key] for r in d.values()], ddof=1):.4f}"
                        for key, _ in OVERALL) + " |")
            L += ["", "按年份（年化超额含成本，同一批种子均值）：", "", "| 组 | " + " | ".join(YEARS) + " |", "|---|" + "---|" * len(YEARS)]
            for k in ("es", "fb10", "b20", "b10", "esyr"):
                d = {s: data[k][s] for s in seeds if s in data[k]}
                if len(d) == len(seeds):
                    L.append(f"| {ARM_NAMES[k]} | " + " | ".join(
                        f"{np.mean([r['years'][y]['ann_excess_w_cost'] for r in d.values()]):+.4f}" for y in YEARS) + " |")
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
        if data["esyr"]:
            chk = [r.get("check_2017_max_abs_diff") for r in data["esyr"].values() if r.get("check_2017_max_abs_diff") is not None]
            if chk:
                L += [f"C 的复现核对：2017 年训练／验证划分与官方相同，其 2017 年预测与官方早停同种子预测的最大绝对差 {max(chk):.2e}（{len(chk)} 个种子）。", ""]
    out = ROOT / "results" / "external" / "protocol_vs_es.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
