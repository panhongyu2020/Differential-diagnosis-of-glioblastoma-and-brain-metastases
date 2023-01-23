import torch
from torchvision import datasets, models, transforms
import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.image as mpig
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np


class getDataset(Dataset):
    def __init__(self, path, img_size, transform, is_training=True):
        super(getDataset, self).__init__()
        self.img_size = img_size
        if is_training:
            self.path = f"./raData/all_tumour/train/{path}"
        else:
            self.path = f"./raData/all_tumour/val/{path}"
        self.WANT_CAT = os.listdir(self.path)
        self.image_files = []
        self.img_labels = []
        self.person_name = []

        labels = os.listdir(self.path)
        for label_name in labels:
            l_path = self.path + "/" + label_name
            for s in os.listdir(l_path):
                # print(s)
                self.person_name.append(s)
                case_path = l_path + "/" + s
                num = len(os.listdir(case_path))
                f = 0
                for img in os.listdir(case_path):
                    if int(num / 2) - 1 < f < int(num / 2) + 1:
                        img_path = case_path + "/" + img
                        self.image_files.append(img_path)
                        self.img_labels.append(self.WANT_CAT.index(label_name))
                    f = f + 1

        assert self.image_files, 'No images found in {}'.format(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        img_path = self.image_files[item]
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)  # resize
        img = np.reshape(img, (self.img_size, self.img_size, 1))
        label = self.img_labels[item]
        return self.transform(img), label


