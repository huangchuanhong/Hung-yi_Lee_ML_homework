import os
from gensim.models import word2vec
import torch

def get_train_label(train_label_path):
    train_list = []
    label_list = []
    with open(train_label_path) as f:
        for line in f.readlines():
            splits = line.strip().split()
            label_list.append(int(splits[0]))
            train_list.append(splits[2:])
    return train_list, label_list

def get_test_data(test_data_path):
    test_list = []
    with open(test_data_path) as f:
        for line in f.readlines():
            splits = line.strip().split(',')[1].strip().split()
            test_list.append(splits)
    return test_list

def get_train_nolabel(train_nolabel_path):
    train_nolabel_list = []
    with open(train_nolabel_path) as f:
        for line in f.readlines():
            train_nolabel_list.append(line.strip().split())
    return train_nolabel_list

def train_word2vec(x, size=250):
    model = word2vec.Word2Vec(x, size=size, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


if __name__ == '__main__':
    root_dir = '../../data'
    train_label_path = os.path.join(root_dir, 'training_label.txt')
    test_data_path = os.path.join(root_dir, 'testing_data.txt')
    train_nolabel_path = os.path.join(root_dir, 'training_nolabel.txt')

    train_list, label_list = get_train_label(train_label_path)
    test_list = get_test_data(test_data_path)
    train_nolabel_list = get_train_nolabel(train_nolabel_path)

    model = train_word2vec(train_list + test_list)
    model.save(os.path.join(root_dir, 'w2v_train_test.model'))
    print('dir(model)={}'.format(dir(model)))

    # embedding = word2vec.Word2Vec.load(os.path.join(root_dir, 'w2v_train_test.model'))
    # embedding_matrix = get_embedding_matrix(embedding)
    # print(embedding_matrix.shape)

    # train_label_path = os.path.join(root_dir, 'training_label.txt')
    # test_data_path = os.path.join(root_dir, 'testing_data.txt')
    # train_nolabel_path = os.path.join(root_dir, 'training_nolabel.txt')
    #
    # train_list, label_list = get_train_label(train_label_path)
    # test_list = get_test_data(test_data_path)
    # train_nolabel_list = get_train_nolabel(train_nolabel_path)
    #
    # model = train_word2vec(train_list + test_list + train_nolabel_list)
    # model.save(os.path.join(root_dir, 'w2v_train_test_nolabel.model'))
    # print('dir(model)={}'.format(dir(model)))