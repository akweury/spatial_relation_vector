# Created by shaji on 02.12.2022
import os
from pathlib import Path
import torch

root = Path(__file__).parents[1]

work_place_path = root / "workplace"
storage = root / "storage"

dataset = storage / "dataset"
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

if __name__ == "__main__":
    print("root path: " + str(root))
    print("storage path: " + str(dataset))
    print("work place path: " + str(work_place_path))
