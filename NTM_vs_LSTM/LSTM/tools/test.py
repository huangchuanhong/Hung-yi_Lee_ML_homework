import sys
sys.path.insert(0, '.')
import random
import torch
from src.apis import test
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--total_epoch', type=int, default=10)
    parser.add_argument('--display_iters', type=int, default=5)
    parser.add_argument('--save_iters', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--test_pth', type=str, default='checkpoints/epoch_2700.pth')
    return parser.parse_args()

args = parse_args()
# sequence_len = random.randint(1, 20)
sequence_len = 7
sequence = torch.empty((1, 0, 8), dtype=torch.float32)
for i in range(sequence_len):
    rand = torch.empty((1, 1, 8)).uniform_(0, 1)
    sequence = torch.cat((sequence, torch.bernoulli(rand)), dim=1)
sequence = torch.cat((sequence, -torch.ones((1, 1, 8), dtype=torch.float32)), dim=1)
test(args, sequence)