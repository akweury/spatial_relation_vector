# Created by shaji at 06-Dec-22

import numpy as np
import cv2 as cv
import json
from PIL import Image, ImageDraw

import torch

import config


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
    return label_data


def load_scaled16bitImage(root, minVal, maxVal):
    img = cv.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535 * (maxVal - minVal) + minVal
    img[np.isnan(img)] = 0
    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img).astype(np.float32)
    return img


def load_32bitImage(root):
    img = cv.imread(root, -1)
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
    return vertex


def vertex_normalization(vertex):
    return vertex


def generate_class_mask(label, classMap, h, w):
    class_mask = np.zeros(shape=(h, w))
    for label_index in range(len(label)):
        # sphere mask
        class_id = None
        mask_i = np.zeros(shape=(h, w))
        shape_attributes = label[label_index]["shape_attributes"]
        if shape_attributes["name"] == "circle":
            x = shape_attributes["cx"]
            y = shape_attributes["cy"]
            r = shape_attributes["r"]
            for y_index in range(int(y - r), int(y + r)):
                chord_length_half = np.sqrt(np.ceil(r) ** 2 - np.abs(y - y_index) ** 2)
                mask_i[y_index, int(np.floor(x - chord_length_half)):int(np.ceil(x + chord_length_half))] = 1
        elif shape_attributes["name"] == "polygon":
            img = Image.new("L", (w, h), 0)
            polygon = []
            for point_index in range(len(shape_attributes["all_points_x"])):
                polygon.append(
                    (shape_attributes["all_points_x"][point_index], shape_attributes["all_points_y"][point_index]))
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            mask_i = np.array(img)

        mask_bool = mask_i != 0
        if label[label_index]["region_attributes"]["classes"] == "sphere":
            class_id = classMap["sphere"]
        elif label[label_index]["region_attributes"]["classes"] == "cube":
            class_id = classMap["cube"]

        if class_id is not None:
            class_mask[mask_bool] = class_id

    return class_mask

#
