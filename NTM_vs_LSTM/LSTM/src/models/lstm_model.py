import torch
from torch.nn import Module
import torch.nn as nn
import numpy as np

class LSTM_Net(Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM_Net, self).__init__()
        self.input_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.input_linear = nn.Linear(hidden_dim, input_dim)
        self.output_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, inputs):
        outputs = torch.empty((inputs.size(0), 0, self.input_dim), dtype=torch.float32, device=inputs.device)
        x, (h0, c0) = self.input_lstm(inputs)
        x = x[:, -1, :]
        x = self.input_linear(x)
        x = x[:, np.newaxis, :]
        for i in range(inputs.size(1) - 1):
            x, _ = self.output_lstm(x, (h0, c0))
            x = self.output_linear(x)
            x = self.sigmoid(x)
            outputs = torch.cat((outputs, x), dim=1)
        return outputs

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    sys.path.insert(0, '.')
    from src.datasets import RandomDatset
    dataset = RandomDatset(2)
    net = LSTM_Net(8, 10, 1)
    for i, inputs in enumerate(dataset):
        if i > 10:
            break
        print('inputs.shape={}'.format(inputs.shape))
        print(net(inputs).shape)

