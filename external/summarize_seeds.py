# -*- coding: utf-8 -*-
"""Summarize ~/ext/runs/<model>/seed_XX.json (written by run_seeds.py) into results/external/.

    .venv/bin/python external/summarize_seeds.py lgb mlp
    .venv/bin/python external/summarize_seeds.py --stem seeds_fixed_budget lstm_fb alstm_fb

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
                    "ann_excess_w_cost": (0.0895, 0.02), "ir_w_cost": (1.1408, 0.23), "mdd_w_cost": (-0.1103, 0.02)},
            # RNNs: README row "Alpha158(with selected 20 features)"
            "gru": {"IC": (0.0315, 0.00), "ICIR": (0.2450, 0.04), "Rank IC": (0.0428, 0.00), "Rank ICIR": (0.3440, 0.03),
                    "ann_excess_w_cost": (0.0344, 0.02), "ir_w_cost": (0.5160, 0.25), "mdd_w_cost": (-0.1017, 0.02)},
            "lstm": {"IC": (0.0318, 0.00), "ICIR": (0.2367, 0.04), "Rank IC": (0.0435, 0.00), "Rank ICIR": (0.3389, 0.03),
                     "ann_excess_w_cost": (0.0381, 0.03), "ir_w_cost": (0.5561, 0.46), "mdd_w_cost": (-0.1207, 0.04)},
            "alstm": {"IC": (0.0362, 0.01), "ICIR": (0.2789, 0.06), "Rank IC": (0.0463, 0.01), "Rank ICIR": (0.3661, 0.05),
                      "ann_excess_w_cost": (0.0470, 0.03), "ir_w_cost": (0.6992, 0.47), "mdd_w_cost": (-0.1072, 0.03)}}
NAMES = {"lgb": "LightGBM", "mlp": "MLP", "gru": "GRU", "lstm": "LSTM", "alstm": "ALSTM",
         "lstm_fb": "LSTM 固定预算", "alstm_fb": "ALSTM 固定预算",
         "lstm_fb5": "LSTM 固定 5 轮", "lstm_fb10": "LSTM 固定 10 轮", "alstm_fb5": "ALSTM 固定 5 轮", "alstm_fb10": "ALSTM 固定 10 轮",
         "lstm_proto": "LSTM 完整协议 20 轮", "alstm_proto": "ALSTM 完整协议 20 轮",
         "lstm_proto10": "LSTM 完整协议 10 轮", "alstm_proto10": "ALSTM 完整协议 10 轮",
         "lstm_esyr": "LSTM 早停＋按年重训", "alstm_esyr": "ALSTM 早停＋按年重训"}
# T17 variants (run_seeds.py VARIANTS): the official config except these two settings and the deployed epoch
NOTES = {m: "与官方配置只差两处：n_epochs 20、early_stop 1000（不触发早停），并部署第 20 轮的参数而非验证最优轮"
            "（`external/fixed_budget.py`）。官方 README 无此设定，故无对照值。" for m in ("lstm_fb", "alstm_fb")}
NOTES.update({f"{m}_fb{e}": f"与官方配置只差：n_epochs {e}（一次 10 轮训练，第 5、10 轮各部署一次）、early_stop 1000，部署第 {e} 轮的参数"
              "（`external/fixed_budget.py`）。" for m in ("lstm", "alstm") for e in (5, 10)})
NOTES.update({f"{m}_proto{t}": f"完整固定预算协议（T18-B{e}，`external/run_yearly.py`）：按年从新训练、扩展窗口，固定 {e} 轮部署最后一轮，"
              "余弦学习率，AdamW 0.03，近期加权抽样（半衰期 504 日）。" + ("只跑种子 0—4（描述性）。" if e == 20 else
              "10 轮参考了 A 的轮数曲线，未取其最优点 5 轮。") for m in ("lstm", "alstm") for t, e in (("", 20), ("10", 10))})
NOTES.update({f"{m}_esyr": "官方早停但按年从新训练、扩展窗口，验证段为训练段末 2 年（T18-C，`external/run_yearly.py`）。"
              for m in ("lstm", "alstm")})


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
    args = sys.argv[1:]
    stem = None
    if args[:1] == ["--stem"]:
        stem, args = args[1], args[2:]
    models = args or ["lgb", "mlp"]
    df = pd.concat([load(m) for m in models], ignore_index=True)
    out = ROOT / "results" / "external"
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or "seeds_" + "_".join(models)
    df.to_csv(out / f"{stem}.csv", index=False, float_format="%.6f")

    L = [f"# 外部复现：{'、'.join(NAMES[m] for m in models)} 多种子（Qlib，沪深300，Alpha158）", "",
         "官方配置不改，只换随机种子；测试期 2017-01-01 至 2020-08-01。逐日预测分数在仓库外 `~/ext/runs/<模型>/seed_XX_pred.pkl`。",
         "年化超额按日超额收益均值 × 252（与 Qlib risk_analysis 相同）；2020 年只到 7 月底，也按 252 年化。", ""]
    for m in models:
        d = df[df.model == m].sort_values("seed")
        n = len(d)
        L += [f"## {NAMES[m]}（{n} 个种子）", "", "### 与官方对照（均值 ± 标准差）", "",
              *([NOTES[m], ""] if m in NOTES else []), "| 指标 | 本次 | 官方 README |", "|---|---|---|"]
        for k in OVERALL:
            o = OFFICIAL.get(m, {}).get(k)
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
