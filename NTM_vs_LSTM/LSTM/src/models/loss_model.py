import torch.nn as nn
from torch.nn import Module

class LossModel(Module):
    def __init__(self):
        super(LossModel, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, labels):
        inputs = inputs.reshape(-1)
        labels = labels.reshape(-1)
        loss = self.bce_loss(inputs, labels)
        return loss