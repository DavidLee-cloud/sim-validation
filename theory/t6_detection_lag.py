# -*- coding: utf-8 -*-
"""T6（论文六）：体制变化的检测滞后。

逐窗 IC 序列标准化后，变化前均值 0、变化后均值 -d（d = 变化幅度 / IC 时间序列标准差 = Delta / s），单位方差。
下界（理论）：即使已知变化时点与变化前均值，单侧 z 检验在 alpha=5% 下达到 50%／80% 检验力所需窗数
  n50 = (1.645/d)^2，n80 = ((1.645+0.842)/d)^2；蓝图中的近似为 (2/d)^2。
实际检测器：单侧 CUSUM（参考值 d/2），阈值按“无变化时平均 100 窗报一次误警”（ARL0≈100，约 8 年）校准，
报告变化从第 1 窗起的平均检测滞后（ADD），与下界对比。实证 d≈0.49（(2/d)^2≈17 窗）。

用法：python theory/t6_detection_lag.py [--quick]
输出：results/theory/t6_detection_lag.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "theory"


def run_length(mean, kref, h, reps, maxT, rng):
    """CUSUM 首次越过阈值的时刻（1 基）；均值为 mean 的单位方差序列，统计量 S=max(0, S - x - kref)。"""
    S = np.zeros(reps)
    rl = np.full(reps, maxT, dtype=float)
    alive = np.ones(reps, dtype=bool)
    for t in range(1, maxT + 1):
        x = mean + rng.standard_normal(reps)
        S = np.maximum(0.0, S - x - kref)
        hit = alive & (S > h)
        rl[hit] = t
        alive &= ~hit
        if not alive.any():
            break
    return rl


def calibrate(kref, reps, rng, target=100.0):
    lo, hi = 0.0, 30.0
    for _ in range(30):
        h = (lo + hi) / 2
        arl = run_length(0.0, kref, h, reps, 3000, rng).mean()
        lo, hi = (h, hi) if arl < target else (lo, h)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 500 if a.quick else 20000
    rng = np.random.default_rng(20261002)
    res = []
    for d in (0.25, 0.33, 0.49, 0.75, 1.0, 1.5):
        kref = d / 2
        h = calibrate(kref, reps, rng)
        arl0 = float(run_length(0.0, kref, h, reps, 3000, rng).mean())
        rl = run_length(-d, kref, h, reps, 3000, rng)
        res.append(dict(d=d, h=h, arl0=arl0, add=float(rl.mean()), median_delay=float(np.median(rl)),
                        q80_delay=float(np.quantile(rl, 0.8)), n50_bound=(1.645 / d) ** 2,
                        n80_bound=((1.645 + 0.842) / d) ** 2, approx_2_over_d_sq=(2 / d) ** 2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("t6_detection_lag_quick.json" if a.quick else "t6_detection_lag.json")).write_text(json.dumps(res, indent=1))
    print("d     ARL0   CUSUM平均滞后  中位  80分位  下界n50  下界n80  (2/d)^2")
    for r in res:
        print(f"{r['d']:<5} {r['arl0']:6.1f}  {r['add']:10.1f}  {r['median_delay']:5.0f}  {r['q80_delay']:6.0f}  "
              f"{r['n50_bound']:7.1f}  {r['n80_bound']:7.1f}  {r['approx_2_over_d_sq']:7.1f}")


if __name__ == "__main__":
    main()
