# Created by shaji on 14-Dec-22
import numpy as np
import torch


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
    def __init__(self, id, color=None, shape=None, pos=None, size=None, material=None):
        self.id = id
        self.color = color
        self.shape = shape
        self.position = pos
        self.size = size
        self.material = material

    def print_info(self):
        print(f"color:{self.color}\n"
              f"color :{self.color}\n"
              f"shape :{self.shape}\n"
              f"pos :{self.position}\n"
              f"size :{self.size}\n"
              f"material :{self.material}\n")


def spatial_obj(id, shape, pos, size):
    return SpatialObject(id, shape=shape, pos=pos, size=size)


def generate_spatial_obj(id, vertex, img, label, mask, categories):
    vertex = vertex.permute(1, 2, 0).numpy()
    img = img.permute(1, 2, 0).numpy()
    mask = mask.squeeze(0).numpy()
    mask[mask > 0.8] = 1
    obj_points = vertex[mask == 1]
    obj_pixels = img[mask == 1]
    center_pos = obj_points.mean(axis=0)
    dim = obj_points.max(axis=0) - obj_points.min(axis=0)
    shape = categories[label]
    color = obj_pixels.mean(axis=0)
    return SpatialObject(id, shape=shape, pos=center_pos, size=dim, color=color)


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


def property_mapping(propertyValues, propertyType):
    mapped_properties = []
    if propertyType == "shape":
        return propertyValues
    if propertyType == "color":
        for propertyValue in propertyValues:
            rgb = ["red", "green", "blue"]
            approx_color = rgb[np.argmax(propertyValue)]
            mapped_properties.append(approx_color)
        return mapped_properties
    if propertyType == "position":

        position_matrix = np.array(propertyValues)
        position_matrix[:, -1] = np.abs(position_matrix[:, -1])
        # normalize
        x_range = position_matrix[:, 0].max() - position_matrix[:, 0].min()
        z_range = position_matrix[:, 2].max() - position_matrix[:, 2].min()
        pos_x = position_matrix[:, 0] / x_range
        pos_z = position_matrix[:, 2] / z_range
        for i in range(position_matrix.shape[0]):
            z_diff = (pos_z[i] - 0.5)
            x_diff = (pos_x[i] - 0.5)
            if z_diff > x_diff:
                if z_diff > np.abs(x_diff):
                    mapped_properties.append("behind")
                else:
                    mapped_properties.append("left")
            else:
                if np.abs(z_diff) > x_diff:
                    mapped_properties.append("front")
                else:
                    mapped_properties.append("right")
        return mapped_properties
    if propertyType == "size":
        size_matrix = np.array(propertyValues)
        volumns = size_matrix.prod(axis=1)
        median_volumn = np.median(volumns)
        for volumn in volumns:
            if volumn > median_volumn:
                mapped_properties.append("big")
            else:
                mapped_properties.append("small")
        return mapped_properties


def calc_property_matrix(objs, propertyNames):
    property_mapped_values_matrix = []
    for propertyType in propertyNames:
        property_values = []
        for obj in objs:
            property_values.append(obj.__dict__[propertyType])
        property_mapped_values_matrix.append(property_mapping(property_values, propertyType))
    property_matrix = []
    for i in range(len(objs)):
        obj_vector = []
        for j in range(len(propertyNames)):
            propertyObj = Property(property_mapped_values_matrix[j][i], propertyNames[j], objs[i].id)
            # propertyObj = {"value": property_mapped_values_matrix[j][i],
            #                "propertyType": propertyNames[j],
            #                # "parentId": objs[i].id,
            #                "commonExist": False}

            obj_vector.append(propertyObj)
        property_matrix.append(obj_vector)
    return property_matrix
