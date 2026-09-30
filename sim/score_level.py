# -*- coding: utf-8 -*-
"""Module B-I: score-level Monte Carlo, no training (design doc section 4.1).

Each draw is one cross-section of ``n`` stocks.  The true expected return is ``mu``; the score is

    s = a * sig + b * z + e

B1  monotonicity (Tweedie): mu = sig, z = 0, e iid log-concave -> E[mu | rank] monotone, no inversion.
B2  inversion: z heavy-tailed (Student-t, df) with an inverted-U contribution to mu in its upper tail,
    while the score loads on z monotonically -> hump-shaped rank curve at the top.
B3  information asymmetry: mu = min(sig, 0) (information only in the lower tail).  Compare top-k by score,
    random k from the pool that survives excluding the bottom by score, and a lower-confidence-bound rule.
"""
from __future__ import annotations

import numpy as np

BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, None))


def _draw(case: str, n: int, rng: np.random.Generator, *, a: float, b: float, noise: float, df: float,
          tail_q: float):
    sig = rng.standard_normal(n)
    if case == "b1":
        e = rng.logistic(size=n) * noise / 1.8138
        return sig, a * sig + e, None
    if case == "b2":
        z = rng.standard_t(df, size=n) if np.isfinite(df) else rng.standard_normal(n)
        thr = np.quantile(z, tail_q)
        # contribution of z to mu: increasing up to thr, flat-to-declining above
        cz = np.where(z <= thr, z, thr - 0.5 * (z - thr))
        cz = (cz - cz.mean()) / (cz.std() + 1e-12)
        mu = sig + 0.5 * cz
        e = noise * rng.standard_normal(n)
        return mu, a * sig + b * z + e, None
    if case == "b3":
        mu = np.minimum(sig, 0.0)
        e = noise * rng.standard_normal(n)
        sd = np.abs(noise * rng.standard_normal(n)) + 0.5 * noise   # per-stock uncertainty for the LCB rule
        return mu, a * sig + e, sd
    raise ValueError(case)


def rank_curve(case: str, *, n: int = 300, draws: int = 20000, a: float = 1.0, b: float = 0.0,
               noise: float = 3.0, df: float = np.inf, tail_q: float = 0.9, k: int = 20, keep: int = 90,
               seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    acc = {f"b{lo}": 0.0 for lo, _ in BUCKETS}
    rules = {"topk": 0.0, "exclude_then_random": 0.0, "lcb": 0.0, "oracle": 0.0}
    for _ in range(draws):
        mu, s, sd = _draw(case, n, rng, a=a, b=b, noise=noise, df=df, tail_q=tail_q)
        mu = mu - mu.mean()
        order = np.argsort(-s)
        for lo, hi in BUCKETS:
            acc[f"b{lo}"] += mu[order[lo:hi]].mean()
        pool = order[:keep]
        rules["topk"] += mu[order[:k]].mean()
        rules["exclude_then_random"] += mu[pool].mean()          # expectation of a random k from the pool
        rules["oracle"] += np.sort(mu)[-k:].mean()
        if sd is not None:
            rules["lcb"] += mu[np.argsort(-(s - sd))[:k]].mean()
    out = {key: v / draws for key, v in acc.items()}
    out.update({f"rule_{key}": v / draws for key, v in rules.items()})
    out.update(case=case, a=a, b=b, noise=noise, df=None if not np.isfinite(df) else df, tail_q=tail_q,
               n=n, draws=draws, k=k, keep=keep, seed=seed)
    return out
