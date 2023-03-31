# Created by shaji on 07-Dec-22
from collections import defaultdict, deque
import datetime
import time

import numpy as np
import cv2 as cv
import json
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist


def load_labels(label_file):
    with open(label_file) as f:
        data = json.load(f)

    # bounding box and mask labels
    label_data = {}
    for image_id in data["_via_image_id_list"]:
        image_data = data["_via_img_metadata"][image_id]
        label_data[image_data["filename"]] = image_data["regions"]

    # classes
    classes = list(data["_via_attributes"]["region"]["classes"]["options"].keys())
    label_data["classes"] = {}
    for i in range(len(classes)):
        label_data["classes"][classes[i]] = i + 1
    classes.insert(0, "background")
    return label_data, classes


def load_scaled16bitImage(root, minVal, maxVal):
    img = cv.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535 * (maxVal - minVal) + minVal
    img[np.isnan(img)] = 0
    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img).astype(np.float32)
    return img


def load_16bitImage(root):
    img = cv.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535
    img[np.isnan(img)] = 0
    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img).astype(np.float32)
    return img


def load_32bitImage(root):
    img = cv.imread(root, -1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img[np.isnan(img)] = 0

    return img


def depth2vertex(depth, K, R, t):
    c, h, w = depth.shape

    camOrig = -R.transpose(0, 1) @ t.unsqueeze(1)
    X = torch.arange(0, depth.size(2)).repeat(depth.size(1), 1) - K[0, 2]
    Y = torch.transpose(torch.arange(0, depth.size(1)).repeat(depth.size(2), 1), 0, 1) - K[1, 2]
    Z = torch.ones(depth.size(1), depth.size(2)) * K[0, 0]
    Dir = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), 2)

    vertex = Dir * (depth.squeeze(0) / torch.norm(Dir, dim=2)).unsqueeze(2).repeat(1, 1, 3)
    vertex = R.transpose(0, 1) @ vertex.permute(2, 0, 1).reshape(3, -1)
    vertex = camOrig.unsqueeze(1).repeat(1, h, w) + vertex.reshape(3, h, w)
    vertex = vertex.permute(1, 2, 0)
    vertex = np.array(vertex)

    mask = (torch.sum(depth, dim=0) == 0)
    vertex[mask] = 0
    return vertex


def generate_class_mask(label, classMap, h, w):
    class_mask = np.zeros(shape=(h, w))
    class_labels = []
    for label_index in range(len(label)):
        # sphere mask
        class_id = None
        mask_i = np.zeros(shape=(h, w))
        shape_attributes = label[label_index]["shape_attributes"]
        img = Image.new("L", (w, h), 0)
        if shape_attributes["name"] == "circle":
            x = shape_attributes["cx"]
            y = shape_attributes["cy"]
            r = shape_attributes["r"]
            ImageDraw.Draw(img).ellipse((int(x - r), int(y - r), int(x + r), int(y + r)), fill=1, outline=1)

            # for y_index in range(int(y - r), int(y + r)):
            #     chord_length_half = np.sqrt(np.ceil(r) ** 2 - np.abs(y - y_index) ** 2)
            #     mask_i[y_index, int(np.floor(x - chord_length_half)):int(np.ceil(x + chord_length_half))] = 1
        elif shape_attributes["name"] in ["polygon", "polyline"]:
            polygon = []
            for point_index in range(len(shape_attributes["all_points_x"])):
                polygon.append(
                    (shape_attributes["all_points_x"][point_index], shape_attributes["all_points_y"][point_index]))
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        mask_i = np.array(img)

        mask_bool = mask_i != 0
        class_label = label[label_index]["region_attributes"]["classes"]
        class_id = classMap[class_label]

        if class_id is not None:
            class_mask[mask_bool] = label_index + 1
            class_labels.append(class_id)

    if int(len(np.unique(class_mask)) - 1) != int(len(class_labels)):
        raise ValueError

    return class_mask, class_labels


def get_masks(args, labelMap, objData, masks, shape):
    h, w = shape[0], shape[1]
    class_mask = np.zeros(shape=(h, w))
    masks = masks[:, :, 0]
    mask_values = np.unique(masks)
    for i in range(len(mask_values)):
        mask_i = masks == mask_values[i]
        class_mask[mask_i] = i

    class_labels = []
    for obj in objData:
        if args.exp == "od":
            class_labels.append(labelMap['obj'][obj["shape"]])
        elif args.exp == "cd":
            class_labels.append(labelMap['color'][obj['material'].split("_")[1]])
        else:
            raise ValueError
    return class_mask, class_labels


def normalize(vertex):
    # all_points = vertex.reshape(-1, 3)
    mask = np.sum(vertex, axis=-1) != 0

    vertex_normalized = np.zeros(shape=vertex.shape)
    valid_min, valid_max = vertex[mask].min(), vertex[mask].max()
    vertex_normalized[mask] = (vertex[mask] - valid_min) / (valid_max - valid_min)

    # check if normalization is correct
    valid_points_recall = np.zeros(shape=vertex_normalized.shape)
    valid_points_recall[mask] = vertex_normalized[mask] * (valid_max - valid_min) + valid_min
    print("recall difference: " + str(np.sum(vertex[mask] - valid_points_recall[mask])))
    assert np.abs(np.sum(vertex[mask] - valid_points_recall[mask])) < 1e-2

    return vertex_normalized, valid_min, valid_max


def load_colors_from_data(obj_list):
    color = torch.zeros(size=[len(obj_list)])
    for o_i, obj in enumerate(obj_list):
        if 'red' in obj['material']:
            color[o_i] = 1
        elif 'green' in obj['material']:
            color[o_i] = 2
        elif 'blue' in obj['material']:
            color[o_i] = 3
        else:
            color[o_i] = 0

    return color


def load_pos_from_data(obj_list):
    positions = torch.zeros(size=[len(obj_list), 3])
    for o_i, obj in enumerate(obj_list):
        positions[o_i] = torch.Tensor(obj['position'])

    return positions
