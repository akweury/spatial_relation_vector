# Created by shaji on 14-Dec-22
import numpy as np
from engine import models


class Property():
    def __init__(self, value, name, parent):
        self.name = name
        self.parent = parent
        self.commonExist = False
        self.value = value

    def __eq__(self, other):
        equality = (self.value == other.value) and (self.name == other.name)
        return equality


class SpatialObject():
    def __init__(self, id, color=None, shape=None, pos=None, size=None, material=None, boxCenter=None, pred=None):
        self.id = id
        self.color = color
        self.shape = shape
        self.position = pos
        self.size = size
        self.material = material
        self.screenPosition = boxCenter
        self.pred = pred

    def print_info(self):
        print(f"color:{self.color}\n"
              f"shape :{self.shape}\n"
              f"size :{self.size}\n"
              f"material :{self.material}\n")


def spatial_obj(id, shape, pos, size):
    return SpatialObject(id, shape=shape, pos=pos, size=size)


def generate_spatial_obj(id, vertex, img, label, mask, categories, color_categories, box, pred):
    vertex = vertex.permute(1, 2, 0).to("cpu").numpy()
    # img = img.permute(1, 2, 0).to("cpu").numpy()
    mask = mask.squeeze(0)
    mask[mask > 0.8] = 1
    obj_points = vertex[mask == 1]
    img = img.permute(1, 2, 0)
    obj_pixels = img[mask == 1]
    center_pos = np.median(obj_points, axis=0)
    if np.isnan(center_pos).sum() > 0:
        return None
    dim = obj_points.shape[0]
    shape = categories[label]
    color = models.model_cd(color_categories, img, mask)
    # pred_color = obj_pixels.mean(axis=0)
    box = box.tolist()
    pred = pred.tolist()
    boxCenter = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return SpatialObject(id, shape=shape, pos=center_pos, size=dim, color=color, boxCenter=boxCenter, pred=pred)


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
    srv[pos_start:pos_end] = objB.position - objA.position  # pos difference
    srv[size_start:size_end] = objB.size - objA.size  # size difference
    srv[color_start:color_end] = objB.color - objA.color  # size difference
    srv[sphere] = attrDiff(objA.shape, objB.shape, "sphere")  # sphere coding
    srv[cube] = attrDiff(objA.shape, objB.shape, "cube")  # cube coding

    return srv
