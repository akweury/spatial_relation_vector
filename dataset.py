# Created by shaji at 06-Dec-22

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


import config
import utils

class SyntheticDataset(Dataset):

    def __init__(self, data_path):
        self.X = None



    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        X = torch.load(self.X[item])

        return X