import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .controller import ForwardController, LSTMController
from .head import ReadHead, WriteHead
from .memory import Memory

class NTM(nn.Module):
    def __init__(self, input_dim, output_dim, memory, controller, heads):
        super(NTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory = memory
        self.controller = controller
        self.heads = heads
        # self.read_head = read_head
        # self.write_head = write_head
        self.controller_dim = self.controller.hidden_dim
        self.unit_size = self.memory.unit_size

        # self.fc = nn.Linear(self.controller_dim + self.unit_size, output_dim)
        self.fc = nn.Linear(self.unit_size, output_dim)
        self.init_r = []

        self.read_heads_count = 0
        for i, head in enumerate(heads):
            if head.is_read_head():
                self.read_heads_count += 1
                # init_r_bias = torch.randn(1, self.unit_size) * 0.01
                init_r_bias = torch.zeros(1, self.unit_size)
                self.register_buffer('read{}_bias'.format(i), init_r_bias.data)


        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def init_state(self, batch_size, device):
        read_biases = [getattr(self, 'read{}_bias'.format(i)) for i in range(self.read_heads_count)]
        init_r = [r.clone().repeat(batch_size, 1).to(device) for r in read_biases]
        controller_state = self.controller.init_state(batch_size)
        heads_state = [head.init_w_pre(batch_size).to(device) for head in self.heads]
        # read_head_state = self.read_head.init_w_pre(batch_size)
        # write_head_state = self.write_head.init_w_pre(batch_size)
        return init_r, controller_state, heads_state

    def forward(self, inputs, pre_state):
        pre_reads, pre_controller_state, pre_heads_state = pre_state
        # pre_read_head_state = pre_read_head_state.to(inputs.device)
        # pre_write_head_state = pre_read_head_state.to(inputs.device)

        if pre_controller_state:
            inp = torch.cat([inputs] + pre_reads, dim=1)
            controller_outputs, controller_states = self.controller(inp, pre_controller_state)
        else:
            inp = inputs
            controller_outputs = self.controller(inp)
            controller_states = None

        reads = []
        heads_states = []
        for head, pre_head_state in zip(self.heads, pre_heads_state):
            if head.is_read_head():
                r, head_state = head(controller_outputs, pre_head_state)
                reads.append(r)
            else:
                head_state = head(controller_outputs, pre_head_state)
            heads_states.append(head_state)

        # r, read_head_state = self.read_head(controller_outputs, pre_read_head_state)
        # write_head_state = self.write_head(controller_outputs, pre_write_head_state)
        # inp2 = torch.cat([controller_outputs] + reads, dim=1)
        inp2 = torch.cat(reads, dim=1)
        o = F.sigmoid(self.fc(inp2))
        state = (reads, controller_states, heads_states)
        return o, state

if __name__ == '__main__':
    ntm = NTM('forward', 1, 8, 100, 20, 128)
    inputs = torch.zeros((1, 2, 8))
    ntm(inputs)