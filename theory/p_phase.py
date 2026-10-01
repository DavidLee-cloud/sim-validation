# -*- coding: utf-8 -*-
"""预测极值论文图 4：顶端倒置的相图（分数层，无训练）。

调用冻结模块 sim.score_level.rank_curve 的 b2 情形（只读使用，不改动）：
分数 s = sig + b * z + 噪声，z 为 t(df) 厚尾成分，其对真值的贡献在上尾（分位 tail_q 以上）转为下降。
网格：尾部厚度 df × 厚尾成分载荷 b × 上尾分位 tail_q。倒置量 = 第 6—10 名均值 - 第 1—5 名均值（>0 即倒置），
另报 第 11—20 名 - 第 1—5 名。

用法：python theory/p_phase.py [--quick] [--chunk C --n-chunks K]
输出：results/theory/p_phase/<序号>.json（单元已存在则跳过）；全部完成后 python theory/p_phase.py --collect
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.score_level import rank_curve  # noqa: E402

OUT = ROOT / "results" / "theory" / "p_phase"
DFS = (3.0, 4.0, 5.0, 6.0, 8.0, 12.0, float("inf"))
BS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TQ = (0.8, 0.9)


def units():
    return list(itertools.product(DFS, BS, TQ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    out = OUT.parent / "p_phase_quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    U = units()
    if a.collect:
        rows = [json.loads((out / f"{i}.json").read_text()) for i in range(len(U)) if (out / f"{i}.json").exists()]
        (out.parent / "p_phase_all.json").write_text(json.dumps(rows, indent=1))
        print(f"{len(rows)}/{len(U)} 单元；倒置量（6—10 名 - 1—5 名），tail_q=0.9：")
        print("df \\ b " + " ".join(f"{b:>6}" for b in BS))
        for df in DFS:
            vals = {r["b"]: r["inv_6_10"] for r in rows if r["tail_q"] == 0.9 and (r["df"] or float("inf")) == df}
            print(f"{df:<7}" + " ".join(f"{vals.get(b, float('nan')):+6.3f}" for b in BS))
        return
    draws = 300 if a.quick else 10000
    for i, (df, b, tq) in enumerate(U):
        if i % a.n_chunks != a.chunk or (out / f"{i}.json").exists():
            continue
        r = rank_curve("b2", n=300, draws=draws, b=b, noise=3.0, df=df, tail_q=tq, seed=1000 + i)
        r["inv_6_10"] = r["b5"] - r["b0"]
        r["inv_11_20"] = r["b10"] - r["b0"]
        (out / f"{i}.json").write_text(json.dumps(r))
        if a.quick and i > 6:
            break


if __name__ == "__main__":
    main()
