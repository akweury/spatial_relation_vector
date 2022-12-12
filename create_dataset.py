# Created by shaji on 06-Dec-22
import os
import json
import glob
import numpy as np
import torch
import argparse
from pprint import pprint

from engine import config, dataset_utils as utils


def data2tensor(data_root, args):
    for sub_name in ["test", "train", "val"]:
        data_path = data_root / sub_name
        if not os.path.exists(str(data_path)):
            raise FileNotFoundError
        if not os.path.exists(str(data_path / "tensor")):
            os.makedirs(str(data_path / "tensor"))

        label_json = data_path / "labels.json"
        if not os.path.exists(label_json):
            raise FileNotFoundError("No labels.json has been found!")
        labels, categories = utils.load_labels(label_json)

        depth_files = np.array(sorted(glob.glob(str(data_path / "*depth0.png"), recursive=True)))
        normal_files = np.array(sorted(glob.glob(str(data_path / "*normal0.png"), recursive=True)))
        data_files = np.array(sorted(glob.glob(str(data_path / "*data0.json"), recursive=True)))
        img_files = np.array(sorted(glob.glob(str(data_path / "*image.png"), recursive=True)))

        for item in range(len(data_files)):
            output_tensor_file = str(data_path / "tensor" / f"{str(item).zfill(5)}.pth.tar")
            if args.clear != "true" and (os.path.exists(output_tensor_file) or not os.path.exists(data_files[item])):
                continue

            with open(data_files[item]) as f:
                data = json.load(f)

            depth = utils.load_scaled16bitImage(depth_files[item],
                                                data['minDepth'],
                                                data['maxDepth'])
            vertex = utils.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                        torch.tensor(data["K"]),
                                        torch.tensor(data["R"]).float(),
                                        torch.tensor(data["t"]).float())
            img = utils.load_32bitImage(img_files[item])
            input_data = np.c_[
                vertex,  # 0,1,2
                img,  # 3,4,5
            ]

            # extract labels from each image file
            label = labels[os.path.basename(img_files[item])]
            class_mask, mask_labels = utils.generate_class_mask(label, labels["classes"], vertex.shape[0],
                                                                vertex.shape[1])
            gt = np.c_[
                np.expand_dims(class_mask, axis=2),  # 0
            ]

            # convert to tensor
            input_tensor = torch.from_numpy(input_data.astype(np.float32)).permute(2, 0, 1)
            gt_tensor = torch.from_numpy(gt).permute(2, 0, 1)

            # save tensors
            training_case = {"input_tensor": input_tensor,
                             "gt_tensor": gt_tensor,
                             "mask_labels": mask_labels,
                             "categories": categories}

            torch.save(training_case, output_tensor_file)
            print(f"File {item + 1}/{len(data_files)} saved as a tensor.")


parser = argparse.ArgumentParser()
parser.add_argument('--clear', type=str, default="false", help='set to true to clear existed tensors')
args = parser.parse_args()
dataset_path = config.dataset / "object_detector"
data2tensor(dataset_path, args)
