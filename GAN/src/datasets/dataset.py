import os
import glob
import torch
import cv2
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class FacesDataset(Dataset):
    def __init__(self, root_dir):
        self.imgpaths = glob.glob(os.path.join(root_dir, '*'))
        self.transform =  transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((64, 64)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )

    def __getitem__(self, idx):
        img = cv2.imread(self.imgpaths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgpaths)


if __name__ == '__main__':
    dataset = FacesDataset('../../faces')
    from torch.utils.data.dataloader import DataLoader
    dataloader = iter(DataLoader(dataset, 2))
    print(next(dataloader).shape)

