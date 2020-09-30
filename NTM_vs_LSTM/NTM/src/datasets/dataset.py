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
        # sequence_len = 5
        rand = torch.empty((self.batch_size, sequence_len, 8), dtype=torch.float32).uniform_(0, 1)
        sequence = torch.bernoulli(rand)
        labels = sequence.clone()
        sequence = torch.cat((sequence,
                              torch.zeros((self.batch_size, 1, 8), dtype=torch.float32)), dim=1)
        sequence = torch.cat((sequence, torch.zeros((self.batch_size, sequence_len+1, 1), dtype=torch.float32)), dim=2)
        sequence[:, -1, -1] = 1
        # sequence = torch.empty((self.batch_size, 0, 8), dtype=torch.float32)
        # for i in range(sequence_len):
        #     rand = torch.empty((self.batch_size, 1, 8)).uniform_(0, 1)
        #     sequence = torch.cat((sequence, torch.bernoulli(rand)), dim=1)
        # labels = sequence.clone()
        # sequence = torch.cat((sequence, torch.zeros((self.batch_size, 1, 8), dtype=torch.float32)), dim=1)
        # sequence = torch.cat((sequence, torch.zeros((self.batch_size, sequence_len + 1, 1), dtype=torch.float32)), dim=2)
        # # sequence = torch.cat((sequence, -torch.ones((self.batch_size, 1, 8), dtype=torch.float32)), dim=1)
        # sequence[:, -1, -1] = 1
        return sequence, labels

    def __len__(self):
        return 100000


# Generator of randomized test sequences
def RandomDataloader(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        yield inp.float(), outp.float()


def RandomDataloader_sequence_9(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 2, batch_size, seq_width + 2)
        inp[1:seq_len+1, :, :seq_width] = seq
        inp[0, :, -2] = 1.0
        inp[seq_len+1, :, seq_width+1] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        yield inp.float(), outp.float()

if __name__ == '__main__':
    # dataset = RandomDataset()
    # print(dataset[0])
    dataloader = RandomDataloader(10, 1, 8, 1, 20)
    for batch_iter, (inp, outp) in enumerate(dataloader):
        print('batch_iter={}'.format(batch_iter))
        print('inp.shape={}'.format(inp.shape))
        print('outp.shape={}'.format(outp.shape))

