# -*- coding: utf-8 -*-
"""T1（论文六）：噪声下耐心早停的选轮分布。

验证损失 V_t = mu_t + sigma * e_t（t = 1..N），mu_t = -delta * (1 - exp(-(t-1)/tau))：
delta > 0 表示真值随训练改善，delta = 0 为平坦（低信噪比极限），delta < 0 为真值变差（过拟合）。
噪声 e_t 为轮间 AR(1)（rho）——同一验证集上相邻轮次的误差相关。
耐心 p 的早停：选中轮次 = 第一个“其后 p 轮内没有新低”的记录点；训练跑满 N 轮仍未停则取全程最低。
理论对照（平坦、独立噪声）：P(E*=1) = 1/(p+1)（第 1 轮是 1..p+1 轮中最小者的概率）。

用法：python theory/t1_early_stop.py [--quick]
输出：results/theory/t1_early_stop.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "theory"


def selected_epochs(V: np.ndarray, p: int) -> np.ndarray:
    """V: (reps, N) 验证损失；返回从 1 计的选中轮次。"""
    reps, N = V.shape
    run_min = np.minimum.accumulate(V, axis=1)
    rec = np.zeros_like(V, dtype=bool)
    rec[:, 0] = True
    rec[:, 1:] = V[:, 1:] < run_min[:, :-1]
    nxt = np.full((reps, N), 10 ** 6, dtype=np.int64)       # 下一个记录点的位置（0 基），无则极大
    for t in range(N - 2, -1, -1):
        nxt[:, t] = np.where(rec[:, t + 1], t + 1, nxt[:, t + 1])
    idx = np.arange(N)
    ok = rec & (nxt - idx > p)                               # 其后 p 轮内无新低
    return ok.argmax(axis=1) + 1                             # 末个记录点总满足条件


def simulate(N, p, delta, rho, reps, rng, sigma=1.0, tau=5.0):
    t = np.arange(N)
    mu = -delta * (1.0 - np.exp(-t / tau))
    e = np.empty((reps, N))
    e[:, 0] = rng.standard_normal(reps)
    s = np.sqrt(1.0 - rho ** 2)
    for j in range(1, N):
        e[:, j] = rho * e[:, j - 1] + s * rng.standard_normal(reps)
    E = selected_epochs(mu + sigma * e, p)
    return dict(p=p, delta=delta, rho=rho, N=N, reps=reps,
                p_e1=float((E == 1).mean()), p_le2=float((E <= 2).mean()), mean=float(E.mean()),
                median=float(np.median(E)), p_e1_theory_flat_iid=1.0 / (p + 1),
                hist=np.bincount(E, minlength=N + 1)[1:].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 2000 if a.quick else 200000
    rng = np.random.default_rng(20261002)
    res = [simulate(30, p, d, r, reps, rng)
           for p, d, r in itertools.product((3, 5, 10), (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0), (0.0, 0.5, 0.9))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("t1_early_stop_quick.json" if a.quick else "t1_early_stop.json")).write_text(json.dumps(res, indent=1))
    print("p  delta  rho  P(E*=1)  理论(平坦独立)  P(E*<=2)  均值")
    for r in res:
        if r["rho"] in (0.0, 0.9) and r["delta"] in (-0.5, 0.0, 0.5, 2.0):
            print(f"{r['p']:<2} {r['delta']:+.1f}  {r['rho']:.1f}  {r['p_e1']:.3f}    {r['p_e1_theory_flat_iid']:.3f}         {r['p_le2']:.3f}    {r['mean']:.2f}")


if __name__ == "__main__":
    main()
