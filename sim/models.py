# -*- coding: utf-8 -*-
"""Models for module A/B (design doc sections 3.1 and 4.2).

A-M1  ``MLPReg``: feature -> H-day return, trained with squared error (GKX-style NN).
A-M2  ``E2ENet``: shared trunk with a score head and a return head.  The decision loss is the negative
      softmax-portfolio return plus a downside-risk term; an optional fidelity term (Huber on the return
      head) mirrors the statistical-fidelity idea in penalty form.  Selection is top-k by score.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class Trunk(nn.Module):
    def __init__(self, k: int, hidden=(32, 16), dropout: float = 0.0):
        super().__init__()
        layers, d = [], k
        for w in hidden:
            layers += [nn.Linear(d, w), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = w
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class MLPReg(nn.Module):
    kind = "reg"

    def __init__(self, k: int):
        super().__init__()
        self.trunk = Trunk(k)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x):                        # x [..., K] -> (score, pred), both [...]
        p = self.head(self.trunk(x)).squeeze(-1)
        return p, p


class E2ENet(nn.Module):
    kind = "e2e"

    def __init__(self, k: int):
        super().__init__()
        self.trunk = Trunk(k)
        self.score = nn.Linear(self.trunk.out_dim, 1)
        self.pred = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.score(z).squeeze(-1), self.pred(z).squeeze(-1)


@dataclass
class LossSpec:
    """Loss of a model on a batch of dates.  x [B, N, K], y [B, N] realised H-day returns."""
    kind: str = "reg"           # reg | e2e
    temp: float = 1.0           # softmax temperature of the e2e portfolio (on standardised scores)
    risk: float = 5.0           # weight of the downside term  mean(relu(-port)^2) / scale
    fidelity: float = 0.0       # weight of the Huber fidelity term on the return head (e2e only)

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s, p = model(x)
        if self.kind == "reg":
            return torch.mean((p - y) ** 2) / 1e-2
        z = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-6)
        w = torch.softmax(z / self.temp, dim=1)
        port = (w * y).sum(dim=1)
        loss = -port.mean() / 0.05 + self.risk * torch.mean(torch.relu(-port) ** 2) / 0.05 ** 2
        if self.fidelity > 0:
            loss = loss + self.fidelity * nn.functional.huber_loss(p, y, delta=0.1) / 1e-2
        return loss


def make_model(kind: str, k: int, seed: int) -> nn.Module:
    torch.manual_seed(int(seed))
    return MLPReg(k) if kind == "reg" else E2ENet(k)


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
