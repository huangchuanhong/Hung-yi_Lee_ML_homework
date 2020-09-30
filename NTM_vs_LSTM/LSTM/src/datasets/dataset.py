from torch.utils.data import Dataset
import torch
import random
import numpy as np

class RandomDataset(Dataset):
    # def __init__(self, batch_size):
    #     self.batch_size = batch_size

    def __init__(self):
        self.batch_size = 1

    def __getitem__(self, index):
        sequence_len = random.randint(1, 20)
        sequence = torch.empty((self.batch_size, 0, 8), dtype=torch.float32)
        for i in range(sequence_len):
            rand = torch.empty((self.batch_size, 1, 8)).uniform_(0, 1)
            sequence = torch.cat((sequence, torch.bernoulli(rand)), dim=1)
        sequence = torch.cat((sequence, -torch.ones((self.batch_size, 1, 8), dtype=torch.float32)), dim=1)
        return sequence

    def __len__(self):
        return 100000

if __name__ == '__main__':
    dataset = RandomDataset()
    print(dataset[0])
