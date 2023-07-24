import torch
import numpy as np

def obj2tensor(obj_predictions, max_obj_num):
    obj_tensor = torch.zeros(size=(max_obj_num, 13))
    for obj_i, obj in enumerate(obj_predictions):
        obj_tensor[obj_i, 0:3] = torch.from_numpy(obj.position)
        if obj.color == "red":
            obj_tensor[obj_i, 3] = 1
        elif obj.color == "green":
            obj_tensor[obj_i, 4] = 1
        elif obj.color == "blue":
            obj_tensor[obj_i, 5] = 1
        if obj.shape == "sphere":
            obj_tensor[obj_i, 6] = 1
        elif obj.shape == "cube":
            obj_tensor[obj_i, 7] = 1
        elif obj.shape == "cone":
            obj_tensor[obj_i, 8] = 1
        elif obj.shape == "cylinder":
            obj_tensor[obj_i, 9] = 1
        obj_tensor[obj_i, 10] = obj.pred

        obj_tensor[obj_i, 11:13] = torch.as_tensor(np.array(obj.screenPosition))
    return obj_tensor
