# -*- coding: utf-8 -*-
"""Quick checks: true IC matches the target, labels are consistent, no look-ahead in training dates."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.dgp import DGPConfig, cs_corr, simulate  # noqa: E402
from sim.protocols import RunSpec, _eligible, backtest, summarize  # noqa: E402
from sim.score_level import rank_curve  # noqa: E402


def test_true_ic_matches_target():
    for ic in (0.05, 0.20):
        p = simulate(DGPConfig(n_days=900, ic=ic, drift="regime", style_share=0.3, tail_df=6.0, seed=1))
        got = np.nanmean(cs_corr(p.mu_h[60:-40], p.r_h[60:-40]))
        assert abs(got - ic) < 0.35 * ic + 0.01, (ic, got)


def test_labels_are_sums_of_daily_returns():
    p = simulate(DGPConfig(n_days=300, seed=2))
    d, h = 100, p.cfg.horizon
    assert np.allclose(p.r_h[d], p.r[d + 1:d + h + 1].sum(0), atol=1e-5)
    assert np.allclose(p.mu_h[d], p.exp_daily[d:d + h].sum(0), atol=1e-6)


def test_no_lookahead():
    spec = RunSpec()
    days = _eligible(1000, spec, 20)
    assert days.max() + 20 <= 1000


def test_backtest_runs_small():
    p = simulate(DGPConfig(n_stocks=100, n_days=700, ic=0.10, seed=3))
    for proto in ("p1", "p3", "p4_504"):
        spec = RunSpec(protocol=proto, start_day=500, epochs=3, dates_per_epoch=40, top_k=10)
        res = backtest(p, spec)
        s = summarize(res, 20)
        assert s["n_decisions"] == len(range(500, 680, 20))
        assert np.isfinite(s["ann_net"])


def test_score_level_monotone_b1():
    out = rank_curve("b1", draws=3000, noise=3.0)
    curve = [out[k] for k in ("b0", "b5", "b10", "b20", "b40", "b80")]
    assert all(x >= y - 0.02 for x, y in zip(curve, curve[1:])), curve


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
