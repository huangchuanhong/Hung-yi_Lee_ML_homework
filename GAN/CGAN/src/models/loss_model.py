import torch
import torch.nn as nn

class LossModel(nn.Module):
    def __init__(self, feats_dim, num_classes, hidden_dim):#feats, classes, labels):
        super(LossModel, self).__init__()
        self.fc1 = nn.Linear(feats_dim + num_classes, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.lrelu = nn.LeakyReLU()

    def forward(self, feats, classes, labels):
        x = self.fc1(torch.cat((feats, classes), dim=1))
        x = self.lrelu(self.bn1(x))
        x = self.sigmoid(self.fc2(x))
        loss = self.bce_loss(x, labels)
        return loss

