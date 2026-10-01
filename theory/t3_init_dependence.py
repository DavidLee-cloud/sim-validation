# -*- coding: utf-8 -*-
"""T3（论文六）：早停解依赖初始化，固定预算＋解耦权重衰减的解与初始化无关（线性、强凸情形）。

数据：y = X beta + eps，两种规模——(d=50, 训练 n=2000) 与参数多于样本的 (d=300, n=200)；验证 500、测试 5000；信噪比以总体 R^2 计（0.01／0.05／0.2）。
模型：线性 w，小批量 SGD（批 64，学习率 0.05），初始化 w0 ~ N(0, 1/d)（与 beta 无关）。
协议：
  ES —— 每轮在验证集上计算 MSE，耐心 5、最多 300 轮，取验证最优轮；
  FB —— 固定 300 轮、学习率余弦退火到 1%，解耦权重衰减 lam（w <- w - lr*(grad + lam*w)），取末轮；lam ∈ {0, 1e-3, 1e-2, 1e-1, 1}。
同一份数据、两个不同初始化（与不同批次顺序），比较：
  相对位移 ||w - w0|| / ||w0||；跨种子解距离 ||w_a - w_b|| / (||w_a|| + ||w_b||) * 2；测试预测的跨种子相关；测试 R^2；选中轮次（ES）。
理论对照：lam > 0 且训练充分时 FB 收敛到唯一的岭解（与初始化无关，跨种子距离→0）；ES 在低信噪比下早停于初始化附近，跨种子距离大。

用法：python theory/t3_init_dependence.py [--quick]
输出：results/theory/t3_init_dependence.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "theory"
NVA, NTE, BS, LR, EP, PAT = 500, 5000, 64, 0.05, 300, 5
D, NTR = 50, 2000


def data(r2, rng):
    beta = rng.standard_normal(D)
    beta *= np.sqrt(r2 / (1 - r2)) / np.linalg.norm(beta)      # 特征方差 1，噪声方差 1 → 总体 R^2 = r2
    mk = lambda n: (lambda X: (X, X @ beta + rng.standard_normal(n)))(rng.standard_normal((n, D)))  # noqa: E731
    return beta, mk(NTR), mk(NVA), mk(NTE)


def train(tr, va, proto, lam, seed, epochs):
    rng = np.random.default_rng(seed)
    X, y = tr
    w0 = rng.standard_normal(D) / np.sqrt(D)
    w = w0.copy()
    best, best_w, best_ep, wait = np.inf, w.copy(), 0, 0
    for ep in range(epochs):
        lr = LR * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * ep / epochs))) if proto == "fb" else LR
        for b in np.array_split(rng.permutation(NTR), max(NTR // BS, 1)):
            g = X[b].T @ (X[b] @ w - y[b]) / len(b)
            w -= lr * (g + lam * w)
        if proto == "es":
            v = np.mean((va[0] @ w - va[1]) ** 2)
            if v < best:
                best, best_w, best_ep, wait = v, w.copy(), ep + 1, 0
            else:
                wait += 1
                if wait >= PAT:
                    break
    if proto == "es":
        w = best_w
    else:
        best_ep = epochs
    return w0, w, best_ep


def cell(r2, proto, lam, reps, rng, epochs, size):
    global D, NTR
    D, NTR = size
    rows = []
    for _ in range(reps):
        beta, tr, va, te = data(r2, rng)
        sa, sb = rng.integers(1 << 30, size=2)
        a0, wa, ea = train(tr, va, proto, lam, sa, epochs)
        b0, wb, eb = train(tr, va, proto, lam, sb, epochs)
        pa, pb = te[0] @ wa, te[0] @ wb
        r2te = lambda p: 1 - np.mean((te[1] - p) ** 2) / np.var(te[1])  # noqa: E731
        rows.append([np.linalg.norm(wa - a0) / np.linalg.norm(a0),
                     2 * np.linalg.norm(wa - wb) / (np.linalg.norm(wa) + np.linalg.norm(wb)),
                     np.corrcoef(pa, pb)[0, 1], (r2te(pa) + r2te(pb)) / 2, (ea + eb) / 2,
                     float(min(ea, eb) <= 2)])
    m = np.array(rows).mean(axis=0)
    return dict(d=D, n_train=NTR, r2=r2, proto=proto, lam=lam, reps=reps, epochs=epochs, rel_disp=m[0], cross_seed_dist=m[1],
                pred_corr=m[2], test_r2=m[3], sel_epoch=m[4], p_sel_le2=m[5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps, epochs = (3, 20) if a.quick else (200, EP)
    rng = np.random.default_rng(20261002)
    res = []
    for size in ((50, 2000), (300, 200)):
        for r2 in (0.01, 0.05, 0.2):
            res.append(cell(r2, "es", 0.0, reps, rng, epochs, size))
            for lam in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
                res.append(cell(r2, "fb", lam, reps, rng, epochs, size))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("t3_init_dependence_quick.json" if a.quick else "t3_init_dependence.json")).write_text(json.dumps(res, indent=1))
    print("d/n       R2    协议 lam     相对位移  跨种子距离  预测相关  测试R2   选中轮  选≤2比例")
    for r in res:
        print(f"{r['d']}/{r['n_train']:<5} {r['r2']:<5} {r['proto']:<4} {r['lam']:<7} {r['rel_disp']:8.3f}  {r['cross_seed_dist']:9.3f}  "
              f"{r['pred_corr']:7.3f}  {r['test_r2']:+.4f}  {r['sel_epoch']:6.1f}  {r['p_sel_le2']:.2f}")


if __name__ == "__main__":
    main()
