# Created by shaji at 06-Dec-22

import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.X = np.array(sorted(glob.glob(str(root / "*pth.tar"), recursive=True)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        X = torch.load(self.X[item])

        mask = X["gt_tensor"][0]
        obj_ids = np.unique(mask)[1:]
        masks = mask.numpy() == obj_ids[:,None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        mask_labels = torch.as_tensor(X["mask_labels"], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        pred_labels = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = mask_labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["pred_labels"] = pred_labels

        return X["input_tensor"][3:], target