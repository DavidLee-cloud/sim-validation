# -*- coding: utf-8 -*-
"""Fixed-budget variants of Qlib's LSTM / ALSTM (T17, T18). Qlib's source is not modified; these are subclasses.

FixedBudgetLSTM / FixedBudgetALSTM (T17, T18-A): Qlib's own fit() unchanged, but the parameters are snapshotted
after every training epoch (`_snapshots`, epoch -> state_dict, from 1) and, after Qlib's fit() has reloaded its
best-validation epoch, the LAST epoch is loaded instead. With a constant learning rate, training is identical up to
any epoch e whatever n_epochs is (checked: the validation curves of a 2-epoch and a 200-epoch run of one seed agree),
so a 10-epoch run also yields the 5-epoch fixed-budget model (`_snapshots[5]`).

ProtocolLSTM / ProtocolALSTM (T18-B): the full fixed-budget protocol of paper six, with its own fit():
- fixed n_epochs, deploy the last epoch (no model selection; the valid segment is only logged);
- AdamW with decoupled weight decay (`weight_decay`, 0.03);
- cosine learning rate per epoch: lr_e = lr0 * (0.01 + 0.99 * 0.5 * (1 + cos(pi * e / (E - 1)))), e = 0..E-1;
- recency weighting by weighted sampling with replacement: a training sample dated a trading days before the last
  training date has weight 2^(-a / half_life); each epoch draws len(train) samples (torch WeightedRandomSampler),
  so the epoch length and the loss scale stay as in Qlib's fit (unlike loss weighting, whose mean weight of ~0.3
  over a 9-12 year window would shrink the effective learning rate);
- no evaluation pass over the training set (Qlib's fit spends ~40% of each epoch on it; it does not affect training).
The per-batch step is Qlib's own train_epoch (masked MSE, gradient clipping 3.0).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from qlib.contrib.model.pytorch_alstm_ts import ALSTM
from qlib.contrib.model.pytorch_lstm_ts import LSTM
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset


class _DeployLastEpoch:
    _net_attr = ""

    def train_epoch(self, data_loader):
        super().train_epoch(data_loader)
        self._epochs_done = getattr(self, "_epochs_done", 0) + 1
        self._snapshots[self._epochs_done] = copy.deepcopy(getattr(self, self._net_attr).state_dict())

    def fit(self, *args, **kwargs):
        self._epochs_done, self._snapshots = 0, {}
        out = super().fit(*args, **kwargs)                 # Qlib's fit, which ends by loading the best-validation epoch
        self.deploy_epoch(self._epochs_done)
        return out

    def deploy_epoch(self, epoch: int) -> None:
        getattr(self, self._net_attr).load_state_dict(self._snapshots[epoch])
        self.logger.info("fixed budget: deployed the parameters of epoch %d (of %d trained), not the best-validation epoch",
                         epoch, self._epochs_done)


class FixedBudgetLSTM(_DeployLastEpoch, LSTM):
    _net_attr = "LSTM_model"


class FixedBudgetALSTM(_DeployLastEpoch, ALSTM):
    _net_attr = "ALSTM_model"


class _Protocol:
    _net_attr = ""

    def __init__(self, *args, adamw_weight_decay=0.03, half_life=504.0, lr_floor=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.adamw_weight_decay, self.half_life, self.lr_floor = adamw_weight_decay, half_life, lr_floor

    def lr_at(self, e: int) -> float:
        E = self.n_epochs
        cos = 0.5 * (1 + math.cos(math.pi * e / (E - 1))) if E > 1 else 1.0
        return self.lr * (self.lr_floor + (1 - self.lr_floor) * cos)

    def recency_weights(self, dl_train) -> np.ndarray:
        dates = dl_train.get_index().get_level_values("datetime").values
        uniq = np.unique(dates)
        age = len(uniq) - 1 - np.searchsorted(uniq, dates)  # trading days before the last training date
        return np.power(2.0, -age / self.half_life)

    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        net = getattr(self, self._net_attr)
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        w = self.recency_weights(dl_train)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(dl_train), replacement=True)
        train_loader = DataLoader(ConcatDataset(dl_train, np.ones(len(dl_train))), batch_size=self.batch_size,
                                  sampler=sampler, num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(ConcatDataset(dl_valid, np.ones(len(dl_valid))), batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.n_jobs, drop_last=False)
        self.train_optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.adamw_weight_decay)
        self.fitted = True
        last_year = w >= 2.0 ** (-252 / self.half_life)          # samples within 252 trading days of the training end
        self.logger.info("protocol: %d samples, recency weight min %.4f, sampling share of the last 252 days %.3f",
                         len(w), w.min(), w[last_year].sum() / w.sum())
        evals_result["valid"] = []
        for e in range(self.n_epochs):
            for g in self.train_optimizer.param_groups:
                g["lr"] = self.lr_at(e)
            self.train_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            evals_result["valid"].append(val_score)
            self.logger.info("protocol epoch %d/%d lr %.6f valid %.6f (log only)", e + 1, self.n_epochs, self.lr_at(e), val_score)
        self.logger.info("protocol: deployed the parameters of epoch %d (last)", self.n_epochs)


class ProtocolLSTM(_Protocol, LSTM):
    _net_attr = "LSTM_model"


class ProtocolALSTM(_Protocol, ALSTM):
    _net_attr = "ALSTM_model"
