import argparse
import sys
sys.path.append('.')
from src.apis.train import train, self_learning_train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_label_path', type=str, default='data/training_label.txt')
    parser.add_argument('--test_data_path', type=str, default='data/testing_data.txt')
    parser.add_argument('--train_nolabel_path', type=str, default='data/training_nolabel.txt')
    parser.add_argument('--wv_model_path', type=str, default='data/w2v_train_test_nolabel.model')
    parser.add_argument('--train_val_ratio', type=float, default=0.5)
    parser.add_argument('--sentence_len', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=250)
    parser.add_argument('--hidden_dim', type=int, default=240)
    parser.add_argument('--lstm_num_layers', type=int ,default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--fix_embedding', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--display_iters', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--self_learning_iters', type=int, default=10)
    parser.add_argument('--nolabel_pos_score_thr', type=float, default=0.98)
    parser.add_argument('--nolabel_neg_score_thr', type=float, default=0.02)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # self_learning_train(args)
    train(args)