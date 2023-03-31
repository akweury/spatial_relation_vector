import torch


def  obj2tensor(obj_predictions):

    obj_tensor = torch.zeros(size=(6, 9))
    for obj_i, obj in enumerate(obj_predictions):
        obj_tensor[obj_i, 0:3] = torch.from_numpy(obj.position)
        obj_tensor[obj_i, 3:6] = torch.from_numpy(obj.color)
        if obj.shape == "sphere":
            obj_tensor[obj_i, 6] = 1
        elif obj.shape == "cube":
            obj_tensor[obj_i, 7] = 1
        obj_tensor[obj_i, 8] = obj.pred
    return obj_tensor
