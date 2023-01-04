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
        self.pos = pos
        self.pos_x = pos
        self.pos_y = pos
        self.pos_z = pos
        self.size = size
        self.material = material

    def print_info(self):
        print(f"color:{self.color}\n"
              f"color :{self.color}\n"
              f"shape :{self.shape}\n"
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
    dim = obj_points.shape[0]
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

def size_mapping(ref_size, obj_size):
    if ref_size > obj_size:
        return "bigger than"
    else:
        return "smaller than"

def dir_mapping(ref_pos_vec, pos_vec):
    dir_vec = ref_pos_vec-pos_vec
    # if np.argmax(dir_vec) == 0:
    if dir_vec[0] < 0:
        return "left of"
    else:
        return "right of"


def property_mapping(propertyValues, propertyType):
    mapped_properties = []
    if propertyType == "shape":
        return propertyValues
    # if propertyType == "color":
    #     rgb = ["red", "green", "blue"]
    #     approx_color = rgb[np.argmax(propertyValues)]
    #     return approx_color


def calc_property_matrix(objs, propertyNames):
    obj_relation_matrix = []
    # discrete property values
    for obj_ref in objs:
        # ref obj
        ref_obj_mapping = []
        for propertyType in propertyNames:
            mapped_property = property_mapping(obj_ref.__dict__[propertyType], propertyType)
            ref_obj_mapping.append(mapped_property)
        ref_obj_vector = []
        for j in range(len(propertyNames)):
            propertyObj = Property(ref_obj_mapping[j], propertyNames[j], obj_ref.id)
            ref_obj_vector.append(propertyObj)

        for obj in objs:
            if obj != obj_ref:
                # relationship
                ref_dir = dir_mapping(obj_ref.pos, obj.pos)
                ref_size = size_mapping(obj_ref.size,obj.size)
                # obj vector
                obj_mapping = []
                for propertyType in propertyNames:
                    mapped_property = property_mapping(obj.__dict__[propertyType], propertyType)
                    obj_mapping.append(mapped_property)
                obj_vector = []
                for j in range(len(propertyNames)):
                    propertyObj = Property(obj_mapping[j], propertyNames[j], obj.id)
                    obj_vector.append(propertyObj)
                obj_relation_matrix.append({
                    "ref": ref_obj_vector,
                    "dir": ref_dir,
                    "size": ref_size,
                    "obj": obj_vector,
                })
    return obj_relation_matrix
