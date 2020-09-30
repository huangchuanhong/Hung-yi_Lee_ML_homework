from torch.utils.data.dataset import Dataset
import torch
import numpy as np

def get_embedding_matrix(wv_model):
    embedding_matrix = []
    for word in wv_model.wv.vocab:
        embedding_matrix.append(wv_model[word])
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    vector = torch.empty(1, wv_model.vector_size)
    embedding_matrix = torch.cat([embedding_matrix, vector], 0)
    vector = torch.empty(1, wv_model.vector_size)
    embedding_matrix = torch.cat([embedding_matrix, vector], 0)
    return embedding_matrix

class SentenceDataset(Dataset):
    def __init__(self, sentence_list, label_list, wv_model, sentence_len=20):
        self.wv_model = wv_model
        self.word_idx_map = {}
        for i, word in enumerate(self.wv_model.wv.vocab):
            self.word_idx_map[word] = i
        self.word_idx_map['<PAD>'] = len(self.word_idx_map)
        self.word_idx_map['<UNK>'] = len(self.word_idx_map)
        self.sentence_idxes = []
        for sentence in sentence_list:
            idxes = []
            for i, word in enumerate(sentence):
                if i >= sentence_len:
                    break
                if word in self.word_idx_map:
                    idxes.append(self.word_idx_map[word])
                else:
                    idxes.append(self.word_idx_map['<UNK>'])
            if len(idxes) < sentence_len:
                for _ in range(len(idxes), sentence_len):
                    idxes.append(self.word_idx_map['<PAD>'])
            self.sentence_idxes.append(idxes)
        self.sentence_idxes = torch.tensor(self.sentence_idxes, dtype=torch.int64)
        self.labels = torch.tensor(label_list, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.sentence_idxes[idx], self.labels[idx]

    def __len__(self):
        return len(self.sentence_idxes)

class SentenceTestDataset(Dataset):
    def __init__(self, sentence_list, wv_model, sentence_len=20):
        self.wv_model = wv_model
        self.word_idx_map = {}
        for i, word in enumerate(self.wv_model.wv.vocab):
            self.word_idx_map[word] = i
        self.word_idx_map['<PAD>'] = len(self.word_idx_map)
        self.word_idx_map['<UNK>'] = len(self.word_idx_map)
        self.sentence_idxes = []
        for sentence in sentence_list:
            idxes = []
            for i, word in enumerate(sentence):
                if i >= sentence_len:
                    break
                if word in self.word_idx_map:
                    idxes.append(self.word_idx_map[word])
                else:
                    idxes.append(self.word_idx_map['<UNK>'])
            if len(idxes) < sentence_len:
                for _ in range(len(idxes), sentence_len):
                    idxes.append(self.word_idx_map['<PAD>'])
            self.sentence_idxes.append(idxes)
        self.sentence_idxes = torch.tensor(self.sentence_idxes, dtype=torch.int64)

    def __getitem__(self, idx):
        return self.sentence_idxes[idx]

    def __len__(self):
        return len(self.sentence_idxes)

if __name__ == '__main__':
    # x = torch.nn.Embedding(10, 2)
    # x.weight = torch.nn.Parameter(torch.tensor(np.array(range(20), dtype=np.float32).reshape((10, 2))))
    # print(x(torch.tensor([[1,2],[3,4]])))

    import os
    from word2vector import get_train_label
    from gensim.models import word2vec

    root_dir = '../data'
    train_label_path = os.path.join(root_dir, 'training_label.txt')
    # test_data_path = os.path.join(root_dir, 'testing_data.txt')
    # train_nolabel_path = os.path.join(root_dir, 'training_nolabel.txt')
    #
    train_list, label_list = get_train_label(train_label_path)
    # test_list = get_test_data(test_data_path)
    # train_nolabel_list = get_train_nolabel(train_nolabel_path)
    #
    # model = train_word2vec(train_list + test_list)
    # model.save(os.path.join(root_dir, 'w2v_train_test.model'))
    # print('dir(model)={}'.format(dir(model)))

    embedding = word2vec.Word2Vec.load(os.path.join(root_dir, 'w2v_train_test.model'))

    dataset = SentenceDataset(train_list, label_list, embedding)
    print(dataset[1])
    print(len(dataset))
    # for i, word in enumerate(embedding.wv.vocab):
    #     if i == 0:
    #         print(embedding[word])
