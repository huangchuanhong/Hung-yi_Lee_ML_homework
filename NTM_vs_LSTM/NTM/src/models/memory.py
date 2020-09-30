import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Memory(nn.Module):
    def __init__(self, batch_size, unit_size, memory_len):
        super(Memory, self).__init__()
        self.register_buffer('mem_bias', torch.zeros(memory_len , unit_size))
        # stdev = 1 / (np.sqrt(memory_len + unit_size))
        # nn.init.uniform_(self.mem_bias, -stdev, stdev)
        # self.register_buffer('memory', torch.zeros((batch_size, memory_len, unit_size)))
        # self.memory = torch.zeros((batch_size, memory_len, unit_size))
        self.batch_size = batch_size
        self.unit_size_ = unit_size
        self.memory_len_ = memory_len

    @property
    def size(self):
        return self.unit_size, self.memory_len

    @property
    def unit_size(self):
        return self.unit_size_

    @property
    def memory_len(self):
        return self.memory_len_

    def reset(self, device):
        # self.memory = torch.zeros((self.batch_size, self.memory_len, self.unit_size)).to(device)
        self.memory = self.mem_bias.clone().repeat(self.batch_size, 1, 1).to(device)

    def read(self, w):
        '''
        :param w: shape (batch_size, memory_len)
        :return: shape (batch_size, unit_size)
        '''
        # r = torch.sum(w.unsqueeze(-1) * self.memory, dim=1)
        r = torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)
        return r

    def write(self, w, e, a):
        '''
        :param w: shape (batch_size, memory_len)
        :param e: shape (batch_size, unit_size)
        :param a: shape (batch_size, unit_size)
        :return:
        '''
        self.prev_mem = self.memory
        self.memory = torch.tensor((self.batch_size, self.memory_len, self.unit_size))
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

        # erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1)) # shape (batch_size, memory_len, unit_size)
        # add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1)) # shape (batch_size, memory_len, unit_size)
        # pre_memory = self.memory
        # self.memory = pre_memory * (1 - erase) + add

    def address(self, k, beta, g, s, gamma, w_pre):
        '''
        :param k: shape (batch_size, unit_size)
        :param beta: scalar, >0
        :param g: scalar, [0, 1]
        :param s: shape (batch_size, 3)
        :param gamma: scalar, >1
        :param w_pre: shape (batch_size, memory_len)
        :return: w, shape (batch_size, memory_len)
        '''
        # wc = F.softmax(beta * F.cosine_similarity(k.unsqueeze(1), self.memory, dim=-1), dim=-1) # shape (batch_size, memory_len)
        wc = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k.unsqueeze(1) + 1e-16, dim=-1), dim=1)
        # print('wc={}'.format(wc))
        # print('max_wc={}'.format(wc.max()))
        # print('argmax_wc={}'.format(wc.argmax()))
        wg = wc * g + (1 - g) * w_pre # shape (batch_size, memory_len)
        wg_pad = torch.cat((wg[:, -1:], wg, wg[:, :1]), dim=1) # shape (batch_size, memory_len + 2)
        ws = F.conv1d(wg_pad.unsqueeze(0), s.unsqueeze(1), groups=wc.size(0)).squeeze(0)
        # w = F.softmax(ws ** gamma, dim=1)
        w = ws ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1)) + 1e-16
        # print('w={}'.format(w))
        # print('max_w={}'.format(w.max()))
        # print('argmax w={}'.format(w.argmax()))
        return w
