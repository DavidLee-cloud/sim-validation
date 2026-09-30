# -*- coding: utf-8 -*-
"""Training protocols P1-P5 and the rolling backtest (design doc sections 3.2-3.3).

Every protocol trains on the same information set: decision day d may use sample days s with
s + H <= d (label fully realised) and s >= ``burn``.  A retrain happens every ``retrain_every`` decisions.

P1   holdout early stopping as in the original chapter protocol: rolling 252-day training window, the
     most recent 120 dates validate (H-day embargo between them), patience 5, <=30 epochs, Adam (L2 1e-5),
     warm start from the previous fit ('p1'); 'p1c' cold start; 'p1x' expanding window with the last 20%
     of dates as validation.
P2   purged blocked K-fold (K=5, H-day purge) picks the epoch count, then refits from scratch on all dates.
P3   fixed budget: 30 epochs, cosine decay to 1%, AdamW weight decay 0.03, fresh init at every retrain.
P4   P3 with recency-weighted date sampling, half-life in days ('p4_252', 'p4_504', 'p4_1008').
P5   ensemble (mean score) of 5 independently initialised P1 members.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .dgp import Panel, cs_corr
from .models import LossSpec, flat_params, make_model

RANK_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, None))


@dataclass
class RunSpec:
    protocol: str = "p3"
    model: str = "reg"               # reg | e2e
    loss: LossSpec = field(default_factory=LossSpec)
    model_seed: int = 0
    start_day: int = 1000
    retrain_every: int = 6
    top_k: int = 20
    epochs: int = 30
    dates_per_epoch: int = 250
    dates_per_batch: int = 8
    lr: float = 1e-3
    burn: int = 60
    val_days: int | None = None      # P1/P1c/P5 validation length in dates; None = 120
    train_window: int = 252          # P1/P1c/P5 rolling training window in dates
    commission: float = 3e-4
    stamp: float = 5e-4


# --------------------------------------------------------------------------- data access
class Data:
    def __init__(self, panel: Panel):
        self.x = torch.from_numpy(panel.x)                    # [T, N, K]
        self.y = torch.from_numpy(np.nan_to_num(panel.r_h))  # [T, N]
        self.h = panel.cfg.horizon

    def batch(self, days: np.ndarray):
        idx = torch.as_tensor(days, dtype=torch.long)
        return self.x[idx], self.y[idx]


def _loss_on(model, data: Data, days: np.ndarray, loss: LossSpec, chunk: int = 32) -> float:
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(days), chunk):
            d = days[i:i + chunk]
            x, y = data.batch(d)
            tot += float(loss(model, x, y)) * len(d)
    model.train()
    return tot / max(1, len(days))


def _train_epochs(model, data: Data, days: np.ndarray, spec: RunSpec, rng: np.random.Generator, *,
                  epochs: int, optimizer: str, weights: np.ndarray | None = None,
                  val_days: np.ndarray | None = None, patience: int | None = None):
    """Returns (best_state, chosen_epoch, val_curve).  Without val_days the last epoch is kept."""
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=1e-5)
        sched = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=0.03)
        steps = epochs * math.ceil(spec.dates_per_epoch / spec.dates_per_batch)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    p = None if weights is None else weights / weights.sum()
    best, best_state, chosen, wait, curve = np.inf, None, epochs, 0, []
    model.train()
    for ep in range(epochs):
        draw = rng.choice(days, size=spec.dates_per_epoch, replace=True, p=p)
        for i in range(0, len(draw), spec.dates_per_batch):
            x, y = data.batch(draw[i:i + spec.dates_per_batch])
            opt.zero_grad(set_to_none=True)
            loss = spec.loss(model, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        if val_days is not None:
            v = _loss_on(model, data, val_days, spec.loss)
            curve.append(v)
            if v < best - 1e-9:
                best, chosen, wait = v, ep + 1, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return chosen, curve


def _eligible(day: int, spec: RunSpec, h: int) -> np.ndarray:
    return np.arange(spec.burn, day - h + 1)


# --------------------------------------------------------------------------- protocols
class Fitter:
    """Holds the model(s) of one run across retrains and applies the protocol."""

    def __init__(self, spec: RunSpec, k: int, data: Data):
        self.spec, self.k, self.data = spec, k, data
        self.rng = np.random.default_rng(10_000 + spec.model_seed)
        self.n_members = 5 if spec.protocol == "p5" else 1
        self.models = [make_model(spec.model, k, 1000 * spec.model_seed + j) for j in range(self.n_members)]
        self.theta0 = [flat_params(m) for m in self.models]
        self.fits = 0

    def _fresh(self, j: int):
        self.models[j] = make_model(self.spec.model, self.k, 1000 * self.spec.model_seed + 97 * (self.fits + 1) + j)

    def fit(self, day: int) -> dict:
        spec, h = self.spec, self.data.h
        days = _eligible(day, spec, h)
        proto = spec.protocol
        info = {"day": int(day), "n_days": int(len(days))}
        t0 = time.perf_counter()
        chosen, disp, disp0 = [], [], []
        for j in range(self.n_members):
            if proto in ("p3",) or proto.startswith("p4") or proto == "p1c" or (proto == "p2"):
                self._fresh(j)
            m = self.models[j]
            start = flat_params(m)
            if proto in ("p1", "p1c", "p1x", "p5"):
                if proto == "p1x":
                    n_val = max(20, int(0.2 * len(days)))
                else:
                    n_val = spec.val_days or 120
                val = days[-n_val:]
                trn = days[: len(days) - n_val - h]
                if proto != "p1x":
                    trn = trn[-spec.train_window:]
                ep, _ = _train_epochs(m, self.data, trn, spec, self.rng, epochs=spec.epochs, optimizer="adam",
                                      val_days=val, patience=5)
            elif proto == "p2":
                ep = self._kfold_epochs(days)
                self._fresh(j)
                m = self.models[j]
                start = flat_params(m)
                _train_epochs(m, self.data, days, spec, self.rng, epochs=ep, optimizer="adam")
            else:
                w = None
                if proto.startswith("p4"):
                    hl = float(proto.split("_")[1])
                    w = 0.5 ** ((days[-1] - days) / hl)
                _train_epochs(m, self.data, days, spec, self.rng, epochs=spec.epochs, optimizer="adamw", weights=w)
                ep = spec.epochs
            end = flat_params(m)
            chosen.append(ep)
            disp.append(float((end - start).norm() / (start.norm() + 1e-12)))
            disp0.append(float((end - self.theta0[j]).norm() / (self.theta0[j].norm() + 1e-12)))
        self.fits += 1
        info.update(chosen_epoch=float(np.mean(chosen)), displacement=float(np.mean(disp)),
                    displacement_from_first_init=float(np.mean(disp0)), fit_sec=round(time.perf_counter() - t0, 2))
        return info

    def _kfold_epochs(self, days: np.ndarray, k: int = 5) -> int:
        spec, h = self.spec, self.data.h
        blocks = np.array_split(days, k)
        curves = []
        for b in blocks:
            lo, hi = b[0] - h, b[-1] + h
            trn = days[(days < lo) | (days > hi)]
            m = make_model(spec.model, self.k, int(self.rng.integers(1 << 30)))
            _, c = _train_epochs(m, self.data, trn, spec, self.rng, epochs=spec.epochs, optimizer="adam", val_days=b)
            curves.append(c)
        return int(np.argmin(np.mean(curves, axis=0)) + 1)

    def score(self, day: int):
        x = self.data.x[day]
        with torch.no_grad():
            outs = [m.eval()(x) for m in self.models]
            for m in self.models:
                m.train()
        s = torch.stack([o[0] for o in outs]).mean(0).numpy()
        p = torch.stack([o[1] for o in outs]).mean(0).numpy()
        return s, p


# --------------------------------------------------------------------------- backtest
def backtest(panel: Panel, spec: RunSpec) -> dict:
    torch.set_num_threads(1)
    data = Data(panel)
    h, t = panel.cfg.horizon, panel.cfg.n_days
    fitter = Fitter(spec, panel.x.shape[-1], data)
    decisions = list(range(spec.start_day, t - h, h))
    fits, rows = [], []
    prev_w: dict[int, float] = {}
    for i, d in enumerate(decisions):
        if i % spec.retrain_every == 0:
            fits.append(fitter.fit(d))
        s, p = fitter.score(d)
        mu, rh = panel.mu_h[d].astype(np.float64), panel.r_h[d].astype(np.float64)
        order = np.argsort(-s, kind="stable")
        top = order[: spec.top_k]
        w = {int(j): 1.0 / spec.top_k for j in top}
        buy = sum(max(w.get(j, 0) - prev_w.get(j, 0), 0) for j in set(w) | set(prev_w))
        sell = sum(max(prev_w.get(j, 0) - w.get(j, 0), 0) for j in set(w) | set(prev_w))
        cost = spec.commission * (buy + sell) + spec.stamp * sell
        prev_w = w
        oracle = np.argsort(-mu)[: spec.top_k]
        # calibration of the return head: slope of realised on predicted (k/c = 1/slope in the dispersion sense)
        pc = p - p.mean()
        slope = float(np.dot(pc, rh - rh.mean()) / (np.dot(pc, pc) + 1e-30))
        slope_true = float(np.dot(pc, mu - mu.mean()) / (np.dot(pc, pc) + 1e-30))
        row = {
            "day": d, "true_ic": float(cs_corr(s[None], mu[None])[0]), "real_ic": float(cs_corr(s[None], rh[None])[0]),
            "top_mu": float(mu[top].mean()), "top_r": float(rh[top].mean()),
            "oracle_mu": float(mu[oracle].mean()), "uni_mu": float(mu.mean()), "uni_r": float(rh.mean()),
            "net": float(rh[top].mean() - cost), "turnover": float(buy + sell),
            "pred_sd": float(p.std()), "slope_real": slope, "slope_true": slope_true,
            "top": [int(j) for j in top],
        }
        for a, b in RANK_BUCKETS:
            sel = order[a:b]
            row[f"b{a}_mu"] = float(mu[sel].mean())
            row[f"b{a}_r"] = float(rh[sel].mean())
        rows.append(row)
    return {"fits": fits, "decisions": rows}


def summarize(res: dict, h: int, from_day: int | None = None) -> dict:
    rows = [r for r in res["decisions"] if from_day is None or r["day"] >= from_day]
    net = np.array([r["net"] for r in rows])
    per_year = 252 / h
    nav = np.concatenate([[1.0], np.cumprod(1 + net)])
    ann = nav[-1] ** (per_year / len(net)) - 1
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    g = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    capture = (g("top_mu") - g("uni_mu")) / (g("oracle_mu") - g("uni_mu") + 1e-12)
    fits = res["fits"]
    return {
        "n_decisions": len(rows), "ann_net": float(ann), "mdd": mdd,
        "sharpe": float(net.mean() / (net.std() + 1e-12) * np.sqrt(per_year)),
        "excess_ann": float(np.mean(net - np.array([r["uni_r"] for r in rows])) * per_year),
        "excess_ir": float(np.mean(net - np.array([r["uni_r"] for r in rows]))
                           / (np.std(net - np.array([r["uni_r"] for r in rows])) + 1e-12) * np.sqrt(per_year)),
        "true_ic": g("true_ic"), "real_ic": g("real_ic"), "real_ic_sd": float(np.std([r["real_ic"] for r in rows])),
        "oracle_capture": float(capture), "top_excess_mu_ann": (g("top_mu") - g("uni_mu")) * per_year,
        "slope_real": g("slope_real"), "slope_true": g("slope_true"),
        "chosen_epoch": float(np.mean([f["chosen_epoch"] for f in fits])),
        "share_epoch_le2": float(np.mean([f["chosen_epoch"] <= 2 for f in fits])),
        "displacement": float(np.mean([f["displacement"] for f in fits])),
        "displacement_from_first_init": float(fits[-1]["displacement_from_first_init"]),
        **{f"bucket{a}_mu_ann": float(np.mean([r[f"b{a}_mu"] - r["uni_mu"] for r in rows]) * per_year)
           for a, _ in RANK_BUCKETS},
        **{f"bucket{a}_r_ann": float(np.mean([r[f"b{a}_r"] - r["uni_r"] for r in rows]) * per_year)
           for a, _ in RANK_BUCKETS},
    }
