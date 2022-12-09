# spatial_relation_vector
The implementation about idea spatial relation vector

---
#### Docker Command

docker build -t ml-sha/srv_docker .

docker run --gpus all -it --rm ml-sha/srv_docker

python3 train.py

---
#### Create training tensors
First, run `create_dataset.py` to calculate masks from dataset images and save them as tensors

---
#### TODO List:
1. train a model using for sphere and cube detection.

