# Created by shaji on 02.12.2022
import os
from pathlib import Path
import torch

root = Path(__file__).parents[1]

work_place_path = root / "workplace"
storage = root / "storage"

dataset = storage / "hide"
output = storage / "output"
models = storage / "models"

if not os.path.exists(str(dataset)):
    os.makedirs(str(dataset))

if not os.path.exists(str(output)):
    os.makedirs(str(output))

if not os.path.exists(str(models)):
    os.makedirs(str(models))

colors = [
    "blue",
    "yellow",
    "red",
    "pink",
    "cyan"
]

categories = ["background", "sphere", "cube"]
color_categories = ["other", "red", "green", "blue"]

# pre-trained model
# model_ball_sphere_detector = models / "od" / "model_best.pth.tar"
model_ball_sphere_detector = models / "od" / "od-checkpoint-49.pth.tar"


propertyNames = ["shape"]

relation_dict = {'size': 'bigger/smaller than',
                 "size_small": 'smaller than',
                 'size_big': 'bigger than'
                 }

if __name__ == "__main__":
    print("root path: " + str(root))
    print("storage path: " + str(dataset))
    # print("work place path: " + str(work_place_path))


