# -*- coding: utf-8 -*-
"""T5（论文六）：硬前 k 选择在分数扰动下的跨种子 Jaccard。

n 只股票的共同分数 s_i（标准正态），两个“种子”各加独立扰动 sigma_p * e_i 后取前 k 只，算两集合的 Jaccard。
近似（理论）：阈值 t 取使 sum_i Phi((s_i - t)/sigma_p) = k 的值，入选概率 pi_i = Phi((s_i - t)/sigma_p)，
E|A∩B| ≈ sum pi_i^2，J ≈ sum pi_i^2 / (2k - sum pi_i^2)。
同时报告“扰动／边界间隔”比：sigma_p ÷ 平均边界间隔 E[s_(k) - s_(k+1)]（实证约 7、Jaccard 约 0.27）。
随机基线：两个独立均匀随机 k 子集的 Jaccard 期望（超几何）。

用法：python theory/t5_topk_jaccard.py [--quick]
输出：results/theory/t5_topk_jaccard.json
"""
import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OUT = Path(__file__).resolve().parent.parent / "results" / "theory"


def j0(n, k):
    t = comb(n, k)
    return sum(comb(k, x) * comb(n - k, k - x) / t * x / (2 * k - x) for x in range(max(0, 2 * k - n), k + 1))


def approx_j(s, k, sp):
    f = lambda t: norm.cdf((s - t) / sp).sum() - k  # noqa: E731
    t = brentq(f, s.min() - 20 * sp - 10, s.max() + 20 * sp + 10)
    pi = norm.cdf((s - t) / sp)
    a = (pi ** 2).sum()
    return a / (2 * k - a)


def cell(n, k, sp, reps, rng):
    js, ja, gaps = [], [], []
    for _ in range(reps):
        s = rng.standard_normal(n)
        o = np.sort(s)[::-1]
        gaps.append(o[k - 1] - o[k])
        A = set(np.argsort(-(s + sp * rng.standard_normal(n)))[:k])
        B = set(np.argsort(-(s + sp * rng.standard_normal(n)))[:k])
        js.append(len(A & B) / len(A | B))
        ja.append(approx_j(s, k, sp))
    g = float(np.mean(gaps))
    return dict(n=n, k=k, sigma_p=sp, reps=reps, mean_gap=g, ratio=sp / g, jaccard_sim=float(np.mean(js)),
                jaccard_approx=float(np.mean(ja)), jaccard_random=j0(n, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 50 if a.quick else 3000
    rng = np.random.default_rng(20261002)
    res = [cell(n, 20, sp, reps, rng)
           for n, sp in itertools.product((50, 92, 300), (0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("t5_topk_jaccard_quick.json" if a.quick else "t5_topk_jaccard.json")).write_text(json.dumps(res, indent=1))
    print("n    sigma_p  扰动/间隔  J模拟  J近似  J随机")
    for r in res:
        print(f"{r['n']:<4} {r['sigma_p']:<7} {r['ratio']:8.2f}  {r['jaccard_sim']:.3f}  {r['jaccard_approx']:.3f}  {r['jaccard_random']:.3f}")


if __name__ == "__main__":
    main()
