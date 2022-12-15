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

def posFromMask(mask, vertex, pos):
    # TODO: pos should be calculated from mask
    return pos


def generate_spatial_obj(vertex, boxes, labels, masks, scores, categories, objects):
    shape = objects["shape"]
    position = posFromMask(masks, vertex, objects["position"])
    size = float(objects["size"])
    return SpatialObject(shape=shape, pos=position,size=size)


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


def calc_srv(objA, objB):
    srv = np.zeros(shape=(6))
    srv[0] = objB.pos[0] - objA.pos[0]  # x axis difference
    srv[1] = objB.pos[1] - objA.pos[1]  # y axis difference
    srv[2] = objB.pos[2] - objA.pos[2]  # z axis difference
    srv[3] = objB.size - objA.size  # size difference
    srv[4] = attrDiff(objA.shape, objB.shape, "sphere")  # sphere coding
    srv[5] = attrDiff(objA.shape, objB.shape, "cube")  # cube coding

    return srv
