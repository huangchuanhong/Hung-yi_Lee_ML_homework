import argparse
import sys
sys.path.insert(0, '.')
from src.apis import stage1_train, stage2_train, stage1_test, stage2_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='data/birds')
    parser.add_argument('-s', '--stage', type=int, default=1)
    parser.add_argument('--s1_batch_size', type=int, default=128)
    parser.add_argument('--s2_batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--total_epoch', type=int, default=600)
    parser.add_argument('--lr_decay_every_epoch', type=int, default=100)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--display_epoch', type=int, default=1)
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--s1_summary_path', type=str, default='output/summary/s1')
    parser.add_argument('--s2_summary_path', type=str, default='output/summary/s2')
    parser.add_argument('--summary_iters', type=int, default=100)
    parser.add_argument('--checkpoint_epoch', type=int, default=1)
    parser.add_argument('--s1_checkpoint_path', type=str, default='output/checkpoints/s1/generator_epoch_380.pth')
    parser.add_argument('--s1_checkpoint_dir', type=str, default='output/checkpoints/s1')
    parser.add_argument('--s2_checkpoint_path', type=str, default=None)
    parser.add_argument('--s2_checkpoint_dir', type=str, default='output/checkpoints/s2')
    parser.add_argument('--s1_test_checkpoint_path', type=str, default='output/checkpoints/s1/generator_epoch_380.pth')
    parser.add_argument('--s2_test_checkpoint_path', type=str, default='output/checkpoints/s2/generator_epoch_280.pth')
    parser.add_argument('--s1_log_path', type=str, default='output/train_log_s1')
    parser.add_argument('--s2_log_path', type=str, default='output/train_log_s2')
    parser.add_argument('--txt_embedding_dim', type=int, default=1024)
    parser.add_argument('--c_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--gf_dim', type=int, default=96)
    parser.add_argument('--df_dim', type=int, default=96)
    return parser.parse_args()

def main(args):
    if not args.test:
        if args.stage == 1:
            stage1_train(args)
        elif args.stage == 2:
            stage2_train(args)
    else:
        if args.stage == 1:
            stage1_test(args)
        elif args.stage == 2:
            stage2_test(args)




if __name__ == '__main__':
    print('main')
    args = parse_args()
    main(args)