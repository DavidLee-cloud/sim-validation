# -*- coding: utf-8 -*-
"""T4（论文六）：近期加权的偏差—方差与最优半衰期（局部水平模型）。

beta_t = beta_{t-1} + q * u_t（随机游走漂移），y_t = beta_t + sigma * e_t。
指数加权估计 b_t = (1-lam) * sum_k lam^k y_{t-k}，lam = 2^(-1/h)（h 为半衰期）。
稳态均方误差（理论）：MSE(lam) = sigma^2 (1-lam)/(1+lam) + q^2 lam^2/(1-lam^2)。
最优 lam 即稳态卡尔曼增益 K = 1 - lam，K^2/(1-K) = q^2/sigma^2。

用法：python theory/t4_halflife.py [--quick]
输出：results/theory/t4_halflife.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

OUT = Path(__file__).resolve().parent.parent / "results" / "theory"


def mse_theory(lam, q, sigma):
    return sigma ** 2 * (1 - lam) / (1 + lam) + q ** 2 * lam ** 2 / (1 - lam ** 2)


def kalman_halflife(q, sigma):
    r = (q / sigma) ** 2
    K = (-r + np.sqrt(r * r + 4 * r)) / 2
    return float(np.log(2) / -np.log(1 - K))


def mse_sim(lam, q, sigma, T, burn, rng):
    beta = np.cumsum(q * rng.standard_normal(T))
    y = beta + sigma * rng.standard_normal(T)
    # 递推 b_t = lam b_{t-1} + (1-lam) y_t，以 b_0 = y_0 起步
    b, _ = lfilter([1 - lam], [1, -lam], y, zi=[lam * y[0]])
    return float(np.mean((b[burn:] - beta[burn:]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    T, burn = (20000, 5000) if a.quick else (400000, 20000)
    rng = np.random.default_rng(20261002)
    hs = np.unique(np.round(np.logspace(np.log10(5), np.log10(5000), 31)))
    res = []
    for qs in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        rows = []
        for h in hs:
            lam = 2 ** (-1 / h)
            rows.append(dict(h=float(h), mse_theory=float(mse_theory(lam, qs, 1.0)),
                             mse_sim=mse_sim(lam, qs, 1.0, T, burn, rng)))
        hsim = min(rows, key=lambda r: r["mse_sim"])["h"]
        hthe = min(rows, key=lambda r: r["mse_theory"])["h"]
        res.append(dict(q_over_sigma=qs, h_opt_kalman=kalman_halflife(qs, 1.0), h_opt_theory_grid=hthe,
                        h_opt_sim_grid=hsim, curve=rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("t4_halflife_quick.json" if a.quick else "t4_halflife.json")).write_text(json.dumps(res, indent=1))
    print("q/sigma   最优半衰期：卡尔曼  理论网格  模拟网格   模拟/理论 MSE 最大偏差")
    for r in res:
        dev = max(abs(c["mse_sim"] / c["mse_theory"] - 1) for c in r["curve"])
        print(f"{r['q_over_sigma']:<8}  {r['h_opt_kalman']:9.1f}  {r['h_opt_theory_grid']:8.0f}  {r['h_opt_sim_grid']:8.0f}   {dev:.3f}")


if __name__ == "__main__":
    main()
