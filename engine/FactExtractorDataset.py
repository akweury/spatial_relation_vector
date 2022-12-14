# Created by shaji on 14-Dec-22


import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class FactExtractorDataset(Dataset):

    def __init__(self, root, folder=None):

        if folder is None:
            raise ValueError("Please set the folder name to train/test ")
        self.root = root / folder / "tensor"
        self.X = np.array(sorted(glob.glob(str(self.root / "*pth.tar"), recursive=True)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        X = torch.load(self.X[item])

        return X["input_tensor"][3:]/255.0, item