# -*- coding: utf-8 -*-
"""Synthetic stock panel with known expected returns (design doc section 2).

Daily return of stock i from day t to t+1:

    r[t+1, i] = beta_i * m[t+1]                 market term, no selection information
              + sum_s e[t, i, s] * f[t+1, s]     style term; style premia switch with a Markov regime
              + alpha[t, i] / H                  learnable idiosyncratic signal (linear + nonlinear, drifting)
              + lowvol[t, i] / H                 optional inverted-U low-volatility component (module B-II)
              + eps[t+1, i]                      Student-t noise, heteroscedastic in the volatility feature

Everything that is predictable at t is collected in ``exp_daily[t]``; the evaluation truth for a decision on
day t is ``mu_h[t] = sum_{j=0}^{H-1} exp_daily[t+j]`` and the realised label is ``r_h[t] = sum r[t+1..t+H]``.
The scale of the predictable part is set so that the cross-sectional correlation between ``mu_h`` and ``r_h``
(the true IC) equals ``cfg.ic`` on average.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# feature roles (columns of X)
F_SIZE, F_VOL, F_VALUE = 0, 1, 2
STYLE_FEATURES = (F_SIZE, F_VOL, F_VALUE)

DRIFT_TAU = {"none": np.inf, "rw_slow": 1000.0, "rw_fast": 250.0}


@dataclass
class DGPConfig:
    n_stocks: int = 300
    n_days: int = 3000
    n_feat: int = 11
    horizon: int = 20
    ic: float = 0.05                 # target true IC of the total predictable part
    drift: str = "none"              # none | rw_slow | rw_fast | regime
    style_share: float = 0.0         # share of predictable variance carried by switching style premia
    regime_days: float = 378.0       # mean regime duration (style premia and 'regime' drift)
    nonlin_share: float = 0.5        # share of alpha variance from nonlinear terms
    tail_df: float = np.inf          # Student-t degrees of freedom of eps (inf = normal)
    lowvol_share: float = 0.0        # share of predictable variance from the inverted-U low-vol component
    feat_phi: float = 0.99           # daily AR(1) persistence of features
    idio_vol_h: float = 0.09         # H-day idiosyncratic volatility of a median stock
    market_vol_h: float = 0.06       # H-day market volatility
    style_vol_h: float = 0.02        # H-day volatility of each style factor return per unit exposure
    market_premium_ann: float = 0.06 # annual drift of the market factor (equity premium)
    seed: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tail_df"] = None if not np.isfinite(self.tail_df) else self.tail_df
        return d


@dataclass
class Panel:
    cfg: DGPConfig
    x: np.ndarray          # [T, N, K] float32, cross-sectionally standardised features
    r: np.ndarray          # [T, N]   daily return realised from day t-1 to t (r[0] = 0)
    exp_daily: np.ndarray  # [T, N]   predictable part of r[t+1] known at t
    mu_h: np.ndarray       # [T, N]   true expected H-day return for a decision on day t (nan at the end)
    r_h: np.ndarray        # [T, N]   realised H-day return r[t+1..t+H] (nan at the end)
    regime: np.ndarray     # [T]      style regime label
    beta: np.ndarray       # [N]


def _zscore(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True) + 1e-12
    return (a - m) / s


def _features(cfg: DGPConfig, rng: np.random.Generator) -> np.ndarray:
    t, n, k = cfg.n_days, cfg.n_stocks, cfg.n_feat
    # persistent stock-level component plus AR(1) dynamics; mild cross-feature correlation
    mix = np.eye(k) + 0.2 * rng.standard_normal((k, k)) / np.sqrt(k)
    x = np.empty((t, n, k), dtype=np.float64)
    level = rng.standard_normal((n, k))
    state = rng.standard_normal((n, k))
    s = np.sqrt(1 - cfg.feat_phi ** 2)
    for d in range(t):
        state = cfg.feat_phi * state + s * rng.standard_normal((n, k))
        x[d] = (0.6 * level + 0.8 * state) @ mix
    return _zscore(x, axis=1).astype(np.float32)


def _nonlinear_basis(x: np.ndarray) -> np.ndarray:
    """Nonlinear transforms of the non-style features, standardised per day."""
    a, b, c, d, e = (x[..., j] for j in (3, 4, 5, 6, 7))
    h = np.stack([np.tanh(a * b), np.maximum(c, 0.0), d * d, (e > 1.0).astype(np.float32), np.tanh(2 * a) * c],
                 axis=-1)
    return _zscore(h, axis=1)


def _coef_path(t: int, dim: int, drift: str, regime_days: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance coefficient path [T, dim] with the requested drift."""
    if drift == "regime":
        a, b = rng.standard_normal(dim), rng.standard_normal(dim)
        lab = _markov(t, regime_days, rng)
        return np.where(lab[:, None] == 0, a, b)
    tau = DRIFT_TAU[drift]
    rho = 0.0 if not np.isfinite(tau) else np.exp(-1.0 / tau)
    out = np.empty((t, dim))
    cur = rng.standard_normal(dim)
    for d in range(t):
        if np.isfinite(tau):
            cur = rho * cur + np.sqrt(1 - rho ** 2) * rng.standard_normal(dim)
        out[d] = cur
    return out


def _markov(t: int, mean_days: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_days
    lab = np.empty(t, dtype=np.int8)
    cur = int(rng.integers(2))
    for d in range(t):
        if rng.random() < p:
            cur = 1 - cur
        lab[d] = cur
    return lab


def simulate(cfg: DGPConfig) -> Panel:
    rng = np.random.default_rng(cfg.seed)
    t, n, h = cfg.n_days, cfg.n_stocks, cfg.horizon
    x = _features(cfg, rng)
    x64 = x.astype(np.float64)

    # ---- risk structure: volatility feature drives beta and idiosyncratic volatility ----
    vol_level = x64[:, :, F_VOL].mean(axis=0)
    beta = 1.0 + 0.25 * _zscore(vol_level)
    idio_scale = np.exp(0.3 * x64[:, :, F_VOL])          # [T, N]
    idio_scale /= np.median(idio_scale)

    # ---- realised noise components (daily) ----
    sd_idio = cfg.idio_vol_h / np.sqrt(h)
    if np.isfinite(cfg.tail_df):
        z = rng.standard_t(cfg.tail_df, size=(t, n)) / np.sqrt(cfg.tail_df / (cfg.tail_df - 2))
    else:
        z = rng.standard_normal((t, n))
    eps = sd_idio * idio_scale * z
    m = cfg.market_premium_ann / 252 + cfg.market_vol_h / np.sqrt(h) * rng.standard_normal(t)
    n_style = len(STYLE_FEATURES)
    f_noise = cfg.style_vol_h / np.sqrt(h) * rng.standard_normal((t, n_style))

    # ---- predictable parts, each with unit cross-sectional variance before scaling ----
    lin_coef = _coef_path(t, cfg.n_feat - n_style, cfg.drift, cfg.regime_days, rng)
    lin = np.einsum("tnk,tk->tn", x64[:, :, n_style:], lin_coef)
    nb = _nonlinear_basis(x64)
    nl_coef = _coef_path(t, nb.shape[-1], cfg.drift, cfg.regime_days, rng)
    nl = np.einsum("tnk,tk->tn", nb, nl_coef)
    alpha_u = np.sqrt(1 - cfg.nonlin_share) * _zscore(lin) + np.sqrt(cfg.nonlin_share) * _zscore(nl)
    alpha_u = _zscore(alpha_u)

    regime = _markov(t, cfg.regime_days, rng)
    prem_dir = rng.standard_normal((2, n_style))
    prem_dir[1] = -prem_dir[0] + 0.5 * rng.standard_normal(n_style)   # premia largely reverse across regimes
    style_u = np.einsum("tnk,tk->tn", x64[:, :, list(STYLE_FEATURES)], prem_dir[regime])
    style_u = _zscore(style_u)

    q = (np.argsort(np.argsort(x64[:, :, F_VOL], axis=1), axis=1) + 0.5) / n     # volatility rank in (0,1)
    lowvol_u = _zscore(np.where(q < 0.3, np.sin(np.pi * q / 0.3), 0.0))         # hump inside the lowest 30%

    shares = np.array([1 - cfg.style_share - cfg.lowvol_share, cfg.style_share, cfg.lowvol_share])
    if (shares < -1e-12).any():
        raise ValueError("style_share + lowvol_share must be <= 1")
    pred_u = (np.sqrt(shares[0]) * alpha_u + np.sqrt(shares[1]) * style_u + np.sqrt(shares[2]) * lowvol_u)
    pred_u = _zscore(pred_u)

    # scale: ic = sd_mu / sqrt(sd_mu^2 + noise_var_h)  ->  sd_mu = ic * sqrt(noise_var_h / (1 - ic^2))
    noise_var_h = (np.mean(idio_scale ** 2) * cfg.idio_vol_h ** 2 + np.var(beta) * cfg.market_vol_h ** 2
                   + n_style * cfg.style_vol_h ** 2)
    sd_mu = cfg.ic * np.sqrt(noise_var_h / (1 - cfg.ic ** 2))
    exp_daily = sd_mu / h * pred_u                                   # predictable part of r[t+1] known at t

    style_real = np.einsum("tnk,tk->tn", x64[:, :, list(STYLE_FEATURES)], f_noise)
    r = np.zeros((t, n))
    r[1:] = exp_daily[:-1] + beta * m[1:, None] + style_real[:-1] + eps[1:]

    csum_r = np.vstack([np.zeros((1, n)), np.cumsum(r[1:], axis=0)])       # csum_r[d] = sum r[1..d]
    csum_e = np.vstack([np.zeros((1, n)), np.cumsum(exp_daily, axis=0)])   # csum_e[d] = sum exp[0..d-1]
    r_h = np.full((t, n), np.nan)
    mu_h = np.full((t, n), np.nan)
    r_h[: t - h] = csum_r[h:t] - csum_r[: t - h]
    mu_h[: t - h] = csum_e[h:t] - csum_e[: t - h]
    return Panel(cfg=cfg, x=x, r=r.astype(np.float32), exp_daily=exp_daily.astype(np.float32),
                 mu_h=mu_h.astype(np.float32), r_h=r_h.astype(np.float32), regime=regime, beta=beta)


def cs_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross-sectional Pearson correlation, ignoring nan rows."""
    a = a - np.nanmean(a, axis=-1, keepdims=True)
    b = b - np.nanmean(b, axis=-1, keepdims=True)
    return np.nansum(a * b, axis=-1) / np.sqrt(np.nansum(a * a, axis=-1) * np.nansum(b * b, axis=-1) + 1e-30)
