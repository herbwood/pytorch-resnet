import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision 
from torch.utils.data import Dataset

from utils import unpickle, download_cifar10


class CIFAR10Dataset(Dataset):

    def __init__(self, basepath="cifar-10-batches-py", phase='train', transform=None, download=False):

        self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                      'data_batch_4', 'data_batch_5',]
        self.val_list = ["test_batch"]
        
        self.basepath = basepath
        self.transform = transform
        self.data = []
        self.labels = []

        if download:
            download_cifar10()

        if phase == 'train':
            downloaded_list = self.train_list
        else:
            downloaded_list = self.val_list

        for filename in downloaded_list:
            filepath = os.path.join(self.basepath, filename)
            data_dict = unpickle(filepath)
            self.data.append(data_dict[b'data'])
            self.labels.extend(data_dict[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    dataset = CIFAR10Dataset(download=True)