# Spatial Relation Vector
The implementation about idea "spatial relation vector"

#### 01. Object detection
Mask R-CNN based object detection
![od_example](./demo_img/object_detection_example.png)

#### 02. learning rules
Learning common exist properties of a given batch images. Then consider them as rules.
![learning_rules_example](./demo_img/learning_rules_example.png)

#### 03. scene manipulation
Manipulate the objects to satisfy the unsatisfied rules.
(Visualization is done by Unity.)
![scene_manipulation_example](./demo_img/scene_modification_example.png)


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

