# -*- coding: utf-8 -*-
"""P7 论文 W1—W3 与修正：从 m 个候选中按 s 个种子的开发均值挑最好者，扩大后（新 20 个种子）的缩水。

模型：候选 j 的真实效应 theta_j ~ N(0, tau^2)；每个种子的估计 = theta_j + N(0, sigma^2)（种子间噪声）。
开发估计 X_j = s 个种子均值；按 X 挑最大者，扩大后估计 Y = 另取 20 个新种子的均值。
W1（tau=0）：期望高估 E[X_win - theta_win] = sigma/sqrt(s) * E[max of m 个标准正态]。
W2：被选者 E[theta | X] = B * X，B = tau^2/(tau^2 + sigma^2/s)（正态—正态收缩，独立候选下挑选不改变条件期望）。
W3：期望高估 ≤ delta 所需种子数（tau=0）：s ≥ (sigma * E[max_m] / delta)^2，近似 2 sigma^2 ln m / delta^2。
修正：经验贝叶斯——用 m 个开发估计的离散度估 tau^2（矩估计，截断于 0），对被选者收缩；比较对 Y 的预测误差。
单位：年化百分点；sigma 取 3（实证配对年化差的种子标准差约 3—3.4 pp）。

用法：python theory/p7_selection.py [--quick]
输出：results/theory/p7_selection.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "theory"


def emax(m, rng, n=400000):
    return float(rng.standard_normal((n, m)).max(axis=1).mean())


def cell(m, s, tau, sigma, reps, rng, s_full=20):
    theta = tau * rng.standard_normal((reps, m))
    X = theta + sigma / np.sqrt(s) * rng.standard_normal((reps, m))
    w = X.argmax(axis=1)
    r = np.arange(reps)
    xw, tw = X[r, w], theta[r, w]
    Y = tw + sigma / np.sqrt(s_full) * rng.standard_normal(reps)
    B = tau ** 2 / (tau ** 2 + sigma ** 2 / s) if tau > 0 else 0.0
    slope = float(np.polyfit(xw, tw, 1)[0]) if tau > 0 else float("nan")
    t2 = np.maximum(X.var(axis=1, ddof=1) - sigma ** 2 / s, 0.0)          # 经验贝叶斯：每次重复内估 tau^2
    Bhat = t2 / (t2 + sigma ** 2 / s)
    xm = X.mean(axis=1)
    eb = xm + Bhat * (xw - xm)                                            # 向候选均值收缩
    return dict(m=m, s=s, tau=tau, sigma=sigma, reps=reps,
                overest=float((xw - tw).mean()), overest_vs_Y=float((xw - Y).mean()),
                w1_theory=float(sigma / np.sqrt(s) * EMAX[m]) if tau == 0 else float("nan"),
                slope_sim=slope, slope_theory=B,
                p_sign_flip=float(((xw > 0) & (Y < 0)).sum() / max((xw > 0).sum(), 1)),
                mse_naive=float(((xw - Y) ** 2).mean()), mse_eb=float(((eb - Y) ** 2).mean()))


EMAX = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 5000 if a.quick else 400000
    rng = np.random.default_rng(20261002)
    ms = (2, 5, 10, 20, 50)
    for m in ms:
        EMAX[m] = emax(m, rng, 20000 if a.quick else 400000)
    res = [cell(m, s, tau, 3.0, reps, rng) for m, s, tau in itertools.product(ms, (1, 3, 7, 9, 20), (0.0, 0.5, 1.0, 2.0))]
    w3 = []
    for m in ms:
        for delta in (0.5, 1.0, 2.0):
            need_theory = (3.0 * EMAX[m] / delta) ** 2
            sim = next((s for s in range(1, 401) if 3.0 / np.sqrt(s) * EMAX[m] <= delta), None)
            w3.append(dict(m=m, delta=delta, s_needed_exact=need_theory, s_needed_approx=2 * 9.0 * np.log(m) / delta ** 2,
                           s_needed_grid=sim))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("p7_selection_quick.json" if a.quick else "p7_selection.json")).write_text(
        json.dumps(dict(cells=res, w3=w3, emax=EMAX), indent=1))
    print("m   s   tau  高估(模拟)  W1理论  斜率模拟/理论  反号比例  MSE 朴素/EB")
    for r in res:
        if r["s"] in (3, 9) and r["tau"] in (0.0, 1.0):
            print(f"{r['m']:<3} {r['s']:<3} {r['tau']:<4} {r['overest']:8.2f}  {r['w1_theory']:7.2f}  "
                  f"{r['slope_sim']:.3f}/{r['slope_theory']:.3f}  {r['p_sign_flip']:.3f}  {r['mse_naive']:.2f}/{r['mse_eb']:.2f}")
    print("W3：m  delta  所需种子（精确／近似 2σ²ln m/δ²）")
    for r in w3:
        print(f"   {r['m']:<3} {r['delta']:<4} {r['s_needed_exact']:7.1f} / {r['s_needed_approx']:7.1f}")


if __name__ == "__main__":
    main()
