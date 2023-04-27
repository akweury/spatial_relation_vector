import torch


def obj2tensor(obj_predictions, max_obj_num):
    obj_tensor = torch.zeros(size=(max_obj_num, 9))
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
        obj_tensor[obj_i, 8] = obj.pred
        print(obj.__dict__)
        obj_tensor[obj_i, 9:11] = torch.tensor(obj.screenPosition)
    return obj_tensor
