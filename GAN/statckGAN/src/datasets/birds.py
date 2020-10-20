import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Birds(Dataset):
    def __init__(self, root_dir, split='train', im_size=64):
        self.root_dir = root_dir
        self.im_size = im_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        split_dir = os.path.join(root_dir, split)
        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embeddings(split_dir)
        self.bbox = self.load_bbox()


    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def load_embeddings(self, data_dir):
        with open(os.path.join(data_dir, 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            print('embeddings: {}'.format(embeddings.shape))
        return embeddings

    def load_bbox(self):
        data_dir = self.root_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.im_size * 76 / 64)
        img = img.resize((load_size, load_size), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        embeddings = self.embeddings[index, :, :]
        # img_path = '/home1/huangchuanhong/datasets/public/mscoco/coco2014/train2014/{}.jpg'.format(key)
        img_path = os.path.join(self.root_dir, 'CUB_200_2011', 'images', key + '.jpg')
        img = self.get_img(img_path, bbox)
        # embedding_idx = random.randint(0, embeddings.shape[0]-1)
        embedding_idx = 0
        embedding = embeddings[embedding_idx, :]
        return img, embedding

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    dataset = Birds('../../data/birds')
    img, embedding = dataset[2]
    print(img.shape)
    print(embedding.shape)
