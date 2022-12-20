# Created by shaji on 14-Dec-22
import numpy as np
import torch


class SpatialObject():
    def __init__(self, color=None, shape=None, pos=None, size=None, material=None):
        self.color = color
        self.shape = shape
        self.pos = pos
        self.size = size
        self.material = material

    def print_info(self):
        print(f"color:{self.color}\n"
              f"color :{self.color}\n"
              f"shape :{self.shape}\n"
              f"pos :{self.pos}\n"
              f"size :{self.size}\n"
              f"material :{self.material}\n")


def spatial_obj(shape, pos, size):
    return SpatialObject(shape=shape, pos=pos, size=size)


def generate_spatial_obj(vertex, img, label, mask, categories):
    vertex = vertex.permute(1, 2, 0).numpy()
    img = img.permute(1, 2, 0).numpy()
    mask = mask.squeeze(0).numpy()
    mask[mask>0.8] = 1
    obj_points = vertex[mask == 1]
    obj_pixels = img[mask == 1]
    center_pos = obj_points.mean(axis=0)
    dim = obj_points.max(axis=0) - obj_points.min(axis=0)
    shape = categories[label]
    color = obj_pixels.mean(axis=0)
    return SpatialObject(shape=shape, pos=center_pos, size=dim, color=color)


def attrDiff(objA, objB, attr):
    if objA == objB:
        if objA == attr:
            return 1
        else:
            return 0
    elif objA == attr:
        return 0.6
    elif objB == attr:
        return 0.3
    else:
        return 0


pos_start = 0
pos_end = 3
size_start = 3
size_end = 6
color_start = 6
color_end = 9
sphere = 9
cube = 10


def calc_srv(objA, objB, entity_num):
    srv = np.zeros(shape=(entity_num))
    srv[pos_start:pos_end] = objB.pos - objA.pos  # pos difference
    srv[size_start:size_end] = objB.size - objA.size  # size difference
    srv[color_start:color_end] = objB.color - objA.color  # size difference
    srv[sphere] = attrDiff(objA.shape, objB.shape, "sphere")  # sphere coding
    srv[cube] = attrDiff(objA.shape, objB.shape, "cube")  # cube coding


    return srv
