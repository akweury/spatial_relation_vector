# Created by shaji on 06-Dec-22
import os
import json
import glob
import numpy as np
import torch

from engine import dataset_utils as utils


def data2tensorManualMask(data_root, args):
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


def data2tensorAutoMask(data_root, args):
    for sub_name in ["test", "train"]:
        data_path = data_root / sub_name
        if not os.path.exists(str(data_path)):
            raise FileNotFoundError
        if not os.path.exists(str(data_path / "tensor")):
            os.makedirs(str(data_path / "tensor"))

        labelFile = str(data_path / 'label.json')
        with open(labelFile) as f:
            labelJson = json.load(f)

        depth_files = np.array(sorted(glob.glob(str(data_path / "*depth0.png"), recursive=True)))
        normal_files = np.array(sorted(glob.glob(str(data_path / "*normal0.png"), recursive=True)))
        mask_files = np.array(sorted(glob.glob(str(data_path / "*mask.png"), recursive=True)))
        data_files = np.array(sorted(glob.glob(str(data_path / "*data0.json"), recursive=True)))
        img_files = np.array(sorted(glob.glob(str(data_path / "*image.png"), recursive=True)))

        for item in range(len(data_files)):
            output_tensor_file = str(data_path / "tensor" / f"{str(item).zfill(5)}.pth.tar")
            img = utils.load_32bitImage(img_files[item])
            # categories = []
            if args.clear != "true" and (os.path.exists(output_tensor_file) or not os.path.exists(data_files[item])):
                continue

            with open(data_files[item]) as f:
                data = json.load(f)

            masks = utils.load_16bitImage(mask_files[item])
            depth = utils.load_scaled16bitImage(depth_files[item],
                                                data['minDepth'],
                                                data['maxDepth'])
            vertex = utils.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                        torch.tensor(data["K"]),
                                        torch.tensor(data["R"]).float(),
                                        torch.tensor(data["t"]).float())

            input_data = np.c_[
                vertex,  # 0,1,2
                img,  # 3,4,5
            ]

            # extract labels from each image file
            objData = data['objects']
            class_mask, mask_labels = utils.get_masks(labelJson, objData, masks, vertex.shape[0],
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
                             "categories": list(labelJson.keys())}

            torch.save(training_case, output_tensor_file)
            print(f"File {item + 1}/{len(data_files)} saved as a tensor.")


def data2tensor_fact_extractor(data_root, args):
    data_path = data_root
    if not os.path.exists(str(data_path)):
        raise FileNotFoundError
    if not os.path.exists(str(data_path / "tensor")):
        os.makedirs(str(data_path / "tensor"))

    depth_files = np.array(sorted(glob.glob(str(data_path / "*depth0.png"), recursive=True)))
    normal_files = np.array(sorted(glob.glob(str(data_path / "*normal0.png"), recursive=True)))
    data_files = np.array(sorted(glob.glob(str(data_path / "*data0.json"), recursive=True)))
    img_files = np.array(sorted(glob.glob(str(data_path / "*image.png"), recursive=True)))

    for item in range(len(data_files)):
        output_tensor_file = str(data_path / "tensor" / f"{str(item).zfill(5)}.pth.tar")
        if args.clear == "false" and (os.path.exists(output_tensor_file) or not os.path.exists(data_files[item])):
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

        vertex_normalized, vertex_min, vertex_max = utils.normalize(vertex)

        img = utils.load_32bitImage(img_files[item])
        input_data = np.c_[
            vertex_normalized,  # 0,1,2
            img,  # 3,4,5
        ]
        # convert to tensor
        input_tensor = torch.from_numpy(input_data.astype(np.float32)).permute(2, 0, 1)

        # save tensors
        training_case = {"input_tensor": input_tensor,
                         "objects": data["objects"],
                         "vertex_max": vertex_max,
                         "vertex_min": vertex_min,
                         'file_name': data_files[item]
                         }
        torch.save(training_case, output_tensor_file)
        print(f"File {item + 1}/{len(data_files)} saved as a tensor.")

# parser = argparse.ArgumentParser()
# parser.add_argument('--clear', type=str, default="false", help='set to true to clear existed tensors')
# parser.add_argument('--dataset', type=str, help='Choose which dataset to be created.')
# args = parser.parse_args()
# dataset_path = config.dataset / args.dataset
# data2tensor(dataset_path, args)
