import sys
sys.path.insert(0, '.')
import random
import torch
from src.apis import test
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='test_log')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model_type', type=str, default='forward')
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--controller_dim', type=int, default=100)
    parser.add_argument('--unit_size', type=int, default=20)
    parser.add_argument('--memory_len', type=int, default=128)
    parser.add_argument('--controller_num_layers', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--total_epoch', type=int, default=10)
    parser.add_argument('--display_iters', type=int, default=200)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--test_pth', type=str, default='checkpoints_forward/iter_99000.pth')#'checkpoints_w2/iter_28000.pth')
    return parser.parse_args()

args = parse_args()
# # # sequence_len = random.randint(1, 20)
sequence_len = 10
# rand = torch.empty((sequence_len, 1, 8), dtype=torch.float32).uniform_(0, 1)
# sequence = torch.bernoulli(rand)
# sequence = torch.cat((torch.zeros((1, 1, 8), dtype=torch.float32),
#                       sequence,
#                       torch.zeros((1, 1, 8), dtype=torch.float32)), dim=0)
# sequence = torch.cat((sequence, torch.zeros((sequence_len+2, 1, 2), dtype=torch.float32)), dim=2)
# sequence[-1, :, -1] = 1
# sequence[0, :, -2] = 1
# test(args, sequence)

rand = torch.empty((sequence_len, 1, 8), dtype=torch.float32).uniform_(0, 1)
sequence = torch.bernoulli(rand)
labels = sequence.clone()
sequence = torch.cat((sequence, torch.zeros((1, 1, 8), dtype=torch.float32)), dim=0)
sequence = torch.cat((sequence, torch.zeros((sequence_len+1, 1, 1), dtype=torch.float32)), dim=2)
sequence[-1, :, -1] = 1
# torch.save(sequence, 'test_sequence.tensor')
test(args, sequence)
