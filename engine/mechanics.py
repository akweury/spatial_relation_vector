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
        return "smaller than, "
    else:
        return "bigger than, "


def dir_mapping(ref_pos_vec, pos_vec):
    dir_vec = pos_vec - ref_pos_vec
    dir_vec[2] = -dir_vec[2]
    rho, phi = cart2pol(dir_vec[0], dir_vec[2])  # only consider x and z axis, ignore y axis
    phi_clock_shift = (90 - int(phi)) % 360
    clock_num_zone = (phi_clock_shift + 15) // 30 % 12

    position = ["north",  # 0
                "northeast", "northeast",  # 1,2
                "east",  # 3
                "southeast", "southeast",  # 4,5
                "south",  # 6
                "southwest", "southwest",  # 7,8
                "west",  # 9
                "northwest", "northwest",  # 10,11
                "north"  # 12
                ]

    return f"in the {position[clock_num_zone]} of, "


def property_mapping(propertyValues, propertyType):
    if propertyType == "shape":
        return propertyValues


def find_ref_obj(objs):
    """
    return the west most object
    """
    ref_obj = objs[0]
    for obj in objs:
        delta_x = obj.position[0] - ref_obj.position[0]
        # new object on the west side
        if delta_x < 0:
            ref_obj = obj
        print("break")

    return ref_obj
