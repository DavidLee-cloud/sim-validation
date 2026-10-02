# -*- coding: utf-8 -*-
"""Summarize ~/ext/runs/<model>/seed_XX.json (written by run_seeds.py) into results/external/.

    .venv/bin/python external/summarize_seeds.py lgb mlp

Writes results/external/seeds_<models>.csv (one row per model x seed, overall and per-year metrics)
and results/external/seeds_<models>.md (per-seed tables, mean/sd vs the Qlib README).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path.home() / "ext" / "runs"
YEARS = ("2017", "2018", "2019", "2020")
OVERALL = ["IC", "ICIR", "Rank IC", "Rank ICIR", "ann_excess_w_cost", "ir_w_cost", "mdd_w_cost",
           "ann_excess_wo_cost", "ir_wo_cost"]
# Qlib examples/benchmarks/README.md, CSI300 Alpha158 table (20-seed mean, sd); annualized return / IR / MDD are with cost
OFFICIAL = {"lgb": {"IC": (0.0448, 0.00), "ICIR": (0.3660, 0.00), "Rank IC": (0.0469, 0.00), "Rank ICIR": (0.3877, 0.00),
                    "ann_excess_w_cost": (0.0901, 0.00), "ir_w_cost": (1.0164, 0.00), "mdd_w_cost": (-0.1038, 0.00)},
            "mlp": {"IC": (0.0376, 0.00), "ICIR": (0.2846, 0.02), "Rank IC": (0.0429, 0.00), "Rank ICIR": (0.3220, 0.01),
                    "ann_excess_w_cost": (0.0895, 0.02), "ir_w_cost": (1.1408, 0.23), "mdd_w_cost": (-0.1103, 0.02)}}
NAMES = {"lgb": "LightGBM", "mlp": "MLP"}


def load(model: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RUNS / model).glob("seed_[0-9]*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        r = {"model": model, "seed": d["seed"], **{k: d[k] for k in OVERALL}}
        for y in YEARS:
            for k, v in d["years"][y].items():
                r[f"{y}_{k}"] = v
        r["fit_sec"], r["total_sec"] = d["fit_sec"], d["total_sec"]
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(x: float, nd: int = 4) -> str:
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def main() -> None:
    models = sys.argv[1:] or ["lgb", "mlp"]
    df = pd.concat([load(m) for m in models], ignore_index=True)
    out = ROOT / "results" / "external"
    out.mkdir(parents=True, exist_ok=True)
    stem = "seeds_" + "_".join(models)
    df.to_csv(out / f"{stem}.csv", index=False, float_format="%.6f")

    L = [f"# 外部复现：{'、'.join(NAMES[m] for m in models)} 多种子（Qlib，沪深300，Alpha158）", "",
         "官方配置不改，只换随机种子；测试期 2017-01-01 至 2020-08-01。逐日预测分数在仓库外 `~/ext/runs/<模型>/seed_XX_pred.pkl`。",
         "年化超额按日超额收益均值 × 252（与 Qlib risk_analysis 相同）；2020 年只到 7 月底，也按 252 年化。", ""]
    for m in models:
        d = df[df.model == m].sort_values("seed")
        n = len(d)
        L += [f"## {NAMES[m]}（{n} 个种子）", "", "### 与官方对照（均值 ± 标准差）", "",
              "| 指标 | 本次 | 官方 README |", "|---|---|---|"]
        for k in OVERALL:
            o = OFFICIAL[m].get(k)
            L.append(f"| {k} | {fmt(d[k].mean())} ± {fmt(d[k].std(ddof=1))}（{fmt(d[k].min())}—{fmt(d[k].max())}） | "
                     + (f"{o[0]:.4f} ± {o[1]:.2f}" if o else "—") + " |")
        L += ["", "### 按年份（各种子均值 ± 标准差）", "",
              "| 年份 | 交易日 | IC | Rank IC | 年化超额（含成本） | 年化超额（不含成本） |", "|---|---|---|---|---|---|"]
        for y in YEARS:
            L.append(f"| {y} | {int(d[f'{y}_n_days'].iloc[0])} | "
                     + " | ".join(f"{fmt(d[f'{y}_{k}'].mean())} ± {fmt(d[f'{y}_{k}'].std(ddof=1))}"
                                  for k in ("IC", "Rank IC", "ann_excess_w_cost", "ann_excess_wo_cost")) + " |")
        L += ["", "### 每个种子（整体与按年份；年化超额均含成本）", "",
              "| 种子 | IC | Rank IC | 年化超额 | 信息比 | " + " | ".join(f"{y} IC | {y} RankIC | {y} 超额" for y in YEARS) + " |",
              "|---|---|---|---|---|" + "---|" * (3 * len(YEARS))]
        for _, r in d.iterrows():
            L.append(f"| {int(r.seed)} | {fmt(r['IC'])} | {fmt(r['Rank IC'])} | {fmt(r['ann_excess_w_cost'])} | {fmt(r['ir_w_cost'], 3)} | "
                     + " | ".join(f"{fmt(r[f'{y}_IC'])} | {fmt(r[f'{y}_Rank IC'])} | {fmt(r[f'{y}_ann_excess_w_cost'])}" for y in YEARS) + " |")
        L += ["", f"用时：每个种子训练 {d.fit_sec.mean():.0f} 秒、训练到分析完 {d.total_sec.mean():.0f} 秒（平均）。", ""]
    (out / f"{stem}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out / (stem + '.csv')} and {out / (stem + '.md')} ({len(df)} rows)")


if __name__ == "__main__":
    main()
