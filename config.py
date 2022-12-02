# Created by shaji at 02.12.2022
from pathlib import Path

root = Path(__file__).parents[0]
dataset = root / "dataset"
work_place_path = root / "workplace"
output_path = root / "output"


# dataset on tp-machine
synthetic_dataset = Path("D:\\UnityProjects\\hide_dataset_unity\\CapturedData\\data_synthetic\\train")


if __name__ == "__main__":
    print("root path: " + str(root))
    print("dataset path: " + str(dataset))
    print("work place path: " + str(work_place_path))
    print("output path: " + str(output_path))
    print("synthetic_dataset: " + str(synthetic_dataset))
