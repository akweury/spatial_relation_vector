# Created by shaji at 02.12.2022
from pathlib import Path

root = Path(__file__).parents[0]
dataset = root / "dataset"
work_place_path = root / "workplace"
output_path = root / "output"


# dataset on tp-machine
left_dataset = dataset / "left"


if __name__ == "__main__":
    print("root path: " + str(root))
    print("dataset path: " + str(dataset))
    print("work place path: " + str(work_place_path))
    print("output path: " + str(output_path))
    print("left_dataset: " + str(left_dataset))
