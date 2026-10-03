# -*- coding: utf-8 -*-
"""T17: fixed-budget variants of Qlib's LSTM / ALSTM that deploy the LAST epoch instead of the best-validation one.

Qlib's fit() always reloads the parameters of the best validation epoch at the end. Without touching Qlib's
source, these subclasses snapshot the parameters after every training epoch (train_epoch) and, after Qlib's own
fit() returns, load the snapshot of the final epoch. Everything else (data, features, optimizer, lr, batch, loss,
validation logging) is Qlib's code unchanged. Used by run_seeds.py as models lstm_fb / alstm_fb, whose configs
change only n_epochs (20) and early_stop (larger than n_epochs, so it never triggers).
"""
from __future__ import annotations

import copy

from qlib.contrib.model.pytorch_alstm_ts import ALSTM
from qlib.contrib.model.pytorch_lstm_ts import LSTM


class _DeployLastEpoch:
    _net_attr = ""

    def train_epoch(self, data_loader):
        super().train_epoch(data_loader)
        self._last_state = copy.deepcopy(getattr(self, self._net_attr).state_dict())
        self._epochs_done = getattr(self, "_epochs_done", 0) + 1

    def fit(self, *args, **kwargs):
        self._epochs_done = 0
        out = super().fit(*args, **kwargs)                 # Qlib's fit, which ends by loading the best-validation epoch
        getattr(self, self._net_attr).load_state_dict(self._last_state)
        self.logger.info("fixed budget: deployed the parameters of epoch %d (last), not the best-validation epoch",
                         self._epochs_done)
        return out


class FixedBudgetLSTM(_DeployLastEpoch, LSTM):
    _net_attr = "LSTM_model"


class FixedBudgetALSTM(_DeployLastEpoch, ALSTM):
    _net_attr = "ALSTM_model"
