# Created by shaji on 04-Jan-23
import math
import numpy as np


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    phi = np.rad2deg(phi)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def size_mapping(ref_size, obj_size):
    if ref_size > obj_size:
        return "bigger than"
    else:
        return "smaller than"


def dir_mapping(ref_pos_vec, pos_vec):
    dir_vec = pos_vec - ref_pos_vec
    dir_vec[2] = -dir_vec[2]
    rho, phi = cart2pol(dir_vec[0], dir_vec[2])  # only consider x and z axis, ignore y axis
    phi_clock_shift = (90 - int(phi)) % 360
    clock_num_zone = (phi_clock_shift + 15) // 30 % 12

    return f"at the {clock_num_zone} o'clock"


def property_mapping(propertyValues, propertyType):
    if propertyType == "shape":
        return propertyValues
    # if propertyType == "color":
    #     rgb = ["red", "green", "blue"]
    #     approx_color = rgb[np.argmax(propertyValues)]
    #     return approx_color
