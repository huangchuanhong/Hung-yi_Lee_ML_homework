import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .controller import ForwardController, LSTMController
from .head import ReadHead, WriteHead
from .memory import Memory
from .ntm import NTM

class EncapsulateNTM(nn.Module):
    def __init__(self, base_model_type, batch_size, input_dim, output_dim, controller_dim, unit_size, memory_len,
                 controller_num_layers=1):
        super(EncapsulateNTM, self).__init__()
        num_heads = 1
        if base_model_type == 'forward':
            controller = ForwardController(input_dim, controller_dim, controller_num_layers)
        elif base_model_type == 'lstm':
            controller = LSTMController(input_dim + unit_size, controller_dim, controller_num_layers)
        memory = Memory(batch_size, unit_size, memory_len)
        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [ReadHead(memory, controller_dim, unit_size),
                      WriteHead(memory, controller_dim, unit_size)]
        # read_head = ReadHead(self.memory, controller_dim, unit_size)
        # write_head = WriteHead(self.memory, controller_dim, unit_size)
        # self.read_linear = nn.Linear(controller_dim + unit_size, output_dim)
        self.ntm = NTM(input_dim, output_dim, memory, controller, heads)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.memory = memory


    def init_sequence(self, batch_size, device):
        self.batch_size = batch_size
        self.memory.reset(device)
        self.previous_state = self.ntm.init_state(batch_size, device)

    def forward(self, inputs=None, device='cuda:0'):
        if inputs is None:
            inputs = torch.zeros((self.batch_size, self.input_dim), dtype=torch.float32, device=device)
        o, self.previous_state = self.ntm(inputs, self.previous_state)
        return o, self.previous_state
