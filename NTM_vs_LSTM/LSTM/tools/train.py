import sys
sys.path.insert(0, '.')
from src.apis import train
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
    parser.add_argument('--load_from', type=str, default='checkpoints/epoch_4300.pth')
    return parser.parse_args()

args = parse_args()
train(args)