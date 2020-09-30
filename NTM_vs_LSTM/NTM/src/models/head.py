import torch.nn as nn
import torch
import torch.nn.functional as F


class HeadBase(nn.Module):
    def __init__(self, memory, controller_dim, unit_size):
        super(HeadBase, self).__init__()
        self.memory = memory
        self.controller_dim = controller_dim
        self.unit_size = unit_size

    def split_outputs(self, outputs):
        cur_point = 0
        splits = []
        for i in self.split_sizes:
            splits.append(outputs[:, cur_point: cur_point + i])
            cur_point = cur_point + i
        return splits

    def init_w_pre(self, batch_size):
        w_pre = torch.zeros((batch_size, self.memory.memory_len))
        w_pre[:, 0] = 1.
        return w_pre

class WriteHead(HeadBase):
    def __init__(self, memory, controller_dim, unit_size):
        super(WriteHead, self).__init__(memory, controller_dim, unit_size)
        # k: unit_size
        # beta, g: 1
        # s: 3
        # gamma: 1
        # e, a: unit_size
        self.fc_write = nn.Linear(controller_dim, 3 * unit_size + 6)
        self.split_sizes = [unit_size, 1, 1, 3, 1, unit_size, unit_size]
        # self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def forward(self, inputs, w_pre):
        outputs = self.fc_write(inputs)
        k, beta, g, s, gamma, e, a = self.split_outputs(outputs)
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s)
        gamma = 1 + F.softplus(gamma)
        e = F.sigmoid(e)
        s_tmp = s.clone()
        s_tmp[s_tmp>0.5] = 1
        s_tmp[s_tmp<=0.5] = 0
        print('s={}'.format(s_tmp))
        # print('k={}'.format(k))
        print('beta={}'.format(beta))
        print('g={}'.format(g))
        # print('s={}'.format(s))
        print('gamma={}'.format (gamma))
        # print('e={}'.format(e))
        # print('a={}'.format(a))
        w = self.memory.address(k, beta, g, s, gamma, w_pre)
        # print('w={}'.format(w))
        print('max_w={}'.format(w.max()))
        print('argmax w={}'.format(w.argmax()))
        print('------------------')
        self.memory.write(w, e, a)
        return w

    def is_read_head(self):
        return False


class ReadHead(HeadBase):
    def __init__(self, memory, controller_dim, unit_size):
        super(ReadHead, self).__init__(memory, controller_dim, unit_size)
        # k: unit_size
        # beta, g: 1
        # s: 3
        # gamma: 1
        self.fc_read = nn.Linear(controller_dim, unit_size + 6)
        self.split_sizes = [unit_size, 1, 1, 3, 1]
        # self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def forward(self, inputs, w_pre):
        outputs = self.fc_read(inputs)
        k, beta, g, s, gamma = self.split_outputs(outputs)
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s)
        gamma = 1 + F.softplus(gamma)
        w = self.memory.address(k, beta, g, s, gamma, w_pre)
        # s_tmp = s.clone()
        # s_tmp[s_tmp>0.5] = 1
        # s_tmp[s_tmp<=0.5] = 0
        # print('s={}'.format(s_tmp))
        # print('max_w={}'.format(w.max()))
        # print('argmax w={}'.format(w.argmax()))
        # print('--------------------')
        return self.memory.read(w), w

    def is_read_head(self):
        return True

