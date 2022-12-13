# Created by shaji on 02.12.2022
from pathlib import Path
import torch

root = Path(__file__).parents[1]

work_place_path = root / "workplace"
storage = root / "storage"
# storage on remote machine
# machinestorage_01 = root / "storage-01" / "storage"
# output_remote = root / "storage-01" / "output"

# storage on tp-machine
dataset = storage / "dataset"
output_local = storage / "output"

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
