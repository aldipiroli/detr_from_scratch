import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super(BaseLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, preds, labels):
        pass


class DetrLoss(BaseLoss):
    def __init__(self, config, logger):
        super(DetrLoss, self).__init__(config, logger)
        self.loss_fn = nn.MSELoss()

    def forward(self, preds, labels):
        loss = self.loss_fn(preds, labels)

        loss_dict = {}
        loss_dict["loss"] = loss
        return loss, loss_dict
