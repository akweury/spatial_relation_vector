# spatial_relation_vector
The implementation about idea spatial relation vector

---
#### Docker Command

docker build -t ml-sha/srv_docker .

docker run --gpus all -it --rm -v /storage-01/ml-jsha:/root/spatial_relation_vector/storage ml-sha/srv_docker

python3 create_dataset.py --clear true --dataset object_detector_big
python3 train.py --exp object_detector_big --machine remote --device gpu --num_epochs 100

###
copy dataset

scp D:\PycharmProjects\spatial_relation_vector\dataset\object_detector.zip ml-jsha@130.83.185.153:/home/ml-jsha/spatial_relation_vector


---
#### Create training tensors
First, run `create_dataset.py` to calculate masks from dataset images and save them as tensors

---
#### TODO List:
1. train a model using for sphere and cube detection.

