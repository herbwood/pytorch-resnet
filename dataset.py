import os
import numpy as np

import torch
from torch.utils import data
import torchvision 
from torch.utils.data import Dataset

from utils import unpickle, download_dataset


class CIFAR10Dataset(Dataset):

    def __init(self, basepath, phase='train', transform=None, download=False):

        train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                      'data_batch_4', 'data_batch_5',]
        val_list = ["test_batch"]

        self.basepath = basepath
        self.transform = transform
        self.data = []
        self.lablels = []

        if download:
            download_dataset()

        if phase == 'train':
            downloaded_list = train_list
        else:
            downloaded_list = val_list

        for filename in downloaded_list:
            filepath = os.path.join(self.basepath, filename)
            data_dict = unpickle(filepath)
            self.data.append(data_dict[b'data'])
            self.labels.append(data_dict[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label