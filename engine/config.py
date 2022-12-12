# Created by shaji on 02.12.2022
from pathlib import Path
import torch

root = Path(__file__).parents[1]

work_place_path = root / "workplace"

# dataset on remote machine
storage_01 = root / "storage-01" / "dataset"
output_remote = root / "storage-01" / "output"

# dataset on tp-machine
dataset = root / "dataset"
output_local = root / "output"

colors = [
    "blue",
    "yellow",
    "red",
    "pink",
    "cyan"
]

if __name__ == "__main__":
    print("root path: " + str(root))
    print("dataset path: " + str(dataset))
    print("work place path: " + str(work_place_path))
