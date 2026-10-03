# -*- coding: utf-8 -*-
"""T15: validation early-stopping epochs of the public Qlib models (read-only, from the 20-seed run logs).

Parses the stdout logs that run_seeds.py wrote under ~/ext (one block per seed, closed by the
"[<model> seed XX] IC=..." line) and, per model and seed, extracts the validation curve, the selected
(best-validation) round counted from 1, and the number of rounds actually trained.

- GRU / LSTM / ALSTM: one round = one epoch; qlib logs "train <score>, valid <score>" per epoch
  (score = -loss) and "best score: ... @ <epoch from 0>"; early stop after `early_stop` epochs without improvement.
- MLP (DNNModelPytorch): no epochs; it trains by mini-batch steps and validates every `eval_steps` steps,
  keeping the step with the lowest validation loss; early stop after `early_stop_rounds` validations without
  improvement. One round = one validation (eval_steps steps); the step and an epoch equivalent are reported too.

    .venv/bin/python external/epoch_stats.py

Writes results/external/epoch_stats.md and results/external/epoch_curves.json (per-seed curves, so the
statistics can be recomputed without the logs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = Path.home() / "ext"
LOGS = {"mlp": ["runs_mlpA.log", "runs_mlpB.log", "runs_mlp_2.log"],
        "gru": ["runs_gru_2.log"], "lstm": ["runs_lstm_2.log"], "alstm": ["runs_alstm_2.log"]}
NAMES = {"mlp": "MLP", "gru": "GRU", "lstm": "LSTM", "alstm": "ALSTM"}
# official Alpha158 configs (examples/benchmarks/<M>/workflow_config_<m>_Alpha158.yaml) and model defaults
LIMITS = {"mlp": "max_steps 8000，每 20 步验证一次（eval_steps 默认 20），连续 50 次验证无改进即停（early_stop_rounds 默认 50）；batch 8192",
          "gru": "n_epochs 200，early_stop 10", "lstm": "n_epochs 200，early_stop 10", "alstm": "n_epochs 200，early_stop 10"}
MAX_ROUNDS = {"mlp": 8000 // 20, "gru": 200, "lstm": 200, "alstm": 200}
PATIENCE = {"mlp": 50, "gru": 10, "lstm": 10, "alstm": 10}
EVAL_STEPS, MLP_BATCH = 20, 8192

PID = re.compile(r"^\[(\d+):MainThread\]")
SUMMARY = re.compile(r"^\[(\w+) seed (\d+)\] IC=")
STEP = re.compile(r"\[Step (\d+)\]: train_loss [-\d.naninf]+, valid_loss ([-\d.]+)")
EPOCH_SCORE = re.compile(r" - train ([-\d.]+), valid ([-\d.]+)$")
BEST = re.compile(r"best score: ([-\d.]+) @ (\d+)")
STOP = re.compile(r"early stop")


def parse(model: str) -> dict:
    """seed -> record; a block is the lines of the pid that logged last before the seed's summary line."""
    out = {}
    for name in LOGS[model]:
        blocks, last_pid = {}, None
        for line in (EXT / name).read_text(encoding="utf-8", errors="replace").splitlines():
            m = PID.match(line)
            if m:
                last_pid = m.group(1)
                blocks.setdefault(last_pid, []).append(line)
                continue
            s = SUMMARY.match(line)
            if s and s.group(1) == model and last_pid is not None:
                out[int(s.group(2))] = extract(model, blocks.pop(last_pid, []))
    return dict(sorted(out.items()))


def extract(model: str, lines: list) -> dict:
    stopped = any(STOP.search(l) for l in lines)
    if model == "mlp":
        steps, vloss = zip(*[(int(m.group(1)), float(m.group(2))) for l in lines for m in [STEP.search(l)] if m])
        best_i = int(np.argmin(vloss))              # first minimum, as qlib keeps the first strictly lower loss
        return {"curve": "valid_loss", "values": list(vloss), "steps": list(steps), "best_round": best_i + 1,
                "best_step": steps[best_i], "rounds_trained": len(vloss), "early_stopped": stopped}
    vals = [float(m.group(2)) for l in lines for m in [EPOCH_SCORE.search(l)] if m]
    b = [BEST.search(l) for l in lines if BEST.search(l)]
    best0 = int(b[-1].group(2))
    assert best0 == int(np.argmax(vals)), (model, best0, vals)
    return {"curve": "valid_score(-loss)", "values": vals, "best_round": best0 + 1,
            "rounds_trained": len(vals), "early_stopped": stopped}


def main() -> None:
    curves = {m: parse(m) for m in LOGS}
    n_train = None
    meta = ROOT / "results" / "external" / "label_meta.json"      # written by top_buckets.py (train-segment rows)
    if meta.exists():
        n_train = json.loads(meta.read_text(encoding="utf-8")).get("n_train_rows")
    L = ["# 论文六：公开模型的验证早停选中轮次（T15）", "",
         "来源：T14 之后 20 种子运行（`external/run_seeds.py`，官方 Alpha158 配置不改、只换种子）的训练日志，未重跑。",
         "选中轮次 = 验证集最优的那一轮（从 1 计）；实际训练轮数 = 早停时已训练的轮数。逐种子曲线见 `epoch_curves.json`。", ""]
    for m, recs in curves.items():
        sel = np.array([r["best_round"] for r in recs.values()])
        trained = np.array([r["rounds_trained"] for r in recs.values()])
        unit = "次验证（每次 20 步）" if m == "mlp" else "个 epoch"
        L += [f"## {NAMES[m]}（{len(recs)} 个种子）", "",
              f"配置：{LIMITS[m]}。轮的单位：{unit}。", "",
              "| 统计 | 选中轮次 | 实际训练轮数 |", "|---|---|---|",
              f"| 均值 | {sel.mean():.2f} | {trained.mean():.2f} |",
              f"| 中位数 | {np.median(sel):.1f} | {np.median(trained):.1f} |",
              f"| 最小—最大 | {sel.min()}—{sel.max()} | {trained.min()}—{trained.max()} |",
              f"| ≤2 轮占比 | {np.mean(sel <= 2):.0%}（{int(np.sum(sel <= 2))}/{len(sel)}） | — |",
              f"| 第 1 轮即最优 | {np.mean(sel == 1):.0%}（{int(np.sum(sel == 1))}/{len(sel)}） | — |",
              f"| 配置上限 / patience | {MAX_ROUNDS[m]} / {PATIENCE[m]} | 全部早停：{'是' if all(r['early_stopped'] for r in recs.values()) else '否'} |", ""]
        vals, cnts = np.unique(sel, return_counts=True)
        L += ["选中轮次分布：" + "，".join(f"{v} 轮 × {c}" for v, c in zip(vals, cnts)), ""]
        if m == "mlp":
            bs = np.array([r["best_step"] for r in recs.values()])
            L.append(f"选中步数：均值 {bs.mean():.0f}，中位数 {np.median(bs):.0f}，范围 {bs.min()}—{bs.max()}。")
            if n_train:
                ep = bs * MLP_BATCH / n_train
                L.append(f"折合 epoch（选中步数 × 8192 ÷ 训练集 {n_train:,} 行）：均值 {ep.mean():.2f}，中位数 {np.median(ep):.2f}，"
                         f"范围 {ep.min():.2f}—{ep.max():.2f}；≤2 个 epoch 的占比 {np.mean(ep <= 2):.0%}。")
            L.append("")
        L += ["| 种子 | 选中轮次 | 实际训练轮数 | 最优验证值 |", "|---|---|---|---|"]
        for s, r in recs.items():
            best_v = r["values"][r["best_round"] - 1]
            extra = f"（第 {r['best_step']} 步）" if m == "mlp" else ""
            L.append(f"| {s} | {r['best_round']}{extra} | {r['rounds_trained']} | {best_v:.6f} |")
        L.append("")
    out = ROOT / "results" / "external"
    (out / "epoch_stats.md").write_text("\n".join(L), encoding="utf-8")
    (out / "epoch_curves.json").write_text(json.dumps(curves, indent=0), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
