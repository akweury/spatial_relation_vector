# spatial_relation_vector
The implementation about idea spatial relation vector

---
#### Docker Command

docker build -t ml-sha/srv_docker .

docker run --gpus all -it --rm -v /storage-01/ml-jsha:/root/spatial_relation_vector/storage ml-sha/srv_docker


##### object detector
python3 create_dataset.py --clear true --dataset object_detector_big
python3 train.py --exp object_detector_big --machine remote --device gpu --num_epochs 10

##### fact extractor
python3 fact_extractor.py --exp fact_extractor --machine remote --device gpu

###### saved output
cd /storage-01/ml-jsha/output/object_detector_big


###
copy dataset

scp D:\PycharmProjects\spatial_relation_vector\dataset\object_detector.zip ml-jsha@130.83.185.153:/home/ml-jsha/spatial_relation_vector


---
#### Create training tensors
First, run `create_dataset.py` to calculate masks from dataset images and save them as tensors

---
#### TODO List:

