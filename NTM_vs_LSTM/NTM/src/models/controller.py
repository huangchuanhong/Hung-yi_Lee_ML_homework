import torch
import torch.nn as nn
import numpy as np

class ForwardController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(ForwardController, self).__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        for i in range(1, num_layers):
            setattr(self, 'fc{}'.format(i), nn.Linear(hidden_dim, hidden_dim))
        self._num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self._hidden_dim = hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def num_layers(self):
        return self._num_layers

    def init_state(self, batch_size):
        return None

    def forward(self, inputs):
        x = self.fc0(inputs)
        for i in range(1, self.num_layers):
            x = self.sigmoid(x)
            x = getattr(self, 'fc{}'.format(i))(x)
        x = self.tanh(x)
        return x


class LSTMController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMController, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)#, batch_first=True)
        self.input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        # self.lstm_h_bias = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim) * 0.05)
        # self.lstm_c_bias = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim) * 0.05)

        self.register_buffer('lstm_h_bias', torch.zeros(self.num_layers, 1, self.hidden_dim))
        self.register_buffer('lstm_c_bias', torch.zeros(self.num_layers, 1, self.hidden_dim))

        # self.reset_parameters()

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def num_layers(self):
        return self._num_layers

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.input_dim + self.hidden_dim))
                nn.init.uniform_(p, -stdev, stdev)


    def init_state(self, batch_size):
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def forward(self, inputs, pre_state):
        inputs = inputs.unsqueeze(0)
        outputs, state = self.lstm(inputs, pre_state)
        outputs = outputs.squeeze(0)
        return outputs, state