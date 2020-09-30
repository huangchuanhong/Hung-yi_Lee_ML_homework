import argparse
import sys
sys.path.insert(0, '.')
from src.apis import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_path', type=str, default='train_log')
    parser.add_argument('--data_root', type=str, default='faces')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--in_dim', type=int, default=100)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--total_epoch', type=int, default=1000)
    parser.add_argument('--discriminator_iters', type=int, default=1)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--display_iters', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints_0_9')
    parser.add_argument('--save_iters', type=int, default=100)
    parser.add_argument('--test_checkpoint', type=str, default='')
    parser.add_argument('--generator_load_from', type=str, default='')
    parser.add_argument('--discriminator_load_from', type=str, default='')
    return parser.parse_args()

args = parse_args()
train(args)