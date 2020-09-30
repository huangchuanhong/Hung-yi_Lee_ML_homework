import sys
sys.path.insert(0, '.')
from src.apis import train
import argparse
import numpy as np
import random

from src.models import NTM
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='train_log_forward')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model_type', type=str, default='forward')  # 'lstm'
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--controller_dim', type=int, default=100)
    parser.add_argument('--unit_size', type=int, default=20)
    parser.add_argument('--memory_len', type=int, default=128)
    parser.add_argument('--controller_num_layers', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--total_iters', type=int, default=100000)
    parser.add_argument('--display_iters', type=int, default=20)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_forward')
    parser.add_argument('--load_from', type=str, default='')
    return parser.parse_args()

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
args = parse_args()
train(args)


# ntm = NTM('forward', 1, 8, 100, 20, 128).cuda()
# inputs = torch.zeros((1, 10, 8)).cuda()
# outputs = ntm(inputs)
# loss = outputs.sum()
# loss.backward()
# outputs = ntm(inputs)
# loss = outputs.sum()
# loss.backward()
# print(id(ntm.state_dict()['memory.memory']))
# print(id(ntm.state_dict()['read_head.memory.memory']))
# print(id(ntm.state_dict()['write_head.memory.memory']))
