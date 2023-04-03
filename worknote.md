#### ssh connection

ssh ml-jsha@130.83.185.153
ssh ml-jsha@130.83.185.155
ssh ml-jsha@130.83.185.145
ssh ml-jsha@130.83.185.147
ssh ml-jsha@130.83.42.209
ssh ml-jsha@130.83.42.211

###### Build docker

docker build -t ml-sha/hide_dataset_docker .

###### Run docker

docker run --gpus all -it -v /home/ml-jsha/storage:/root/spatial_relation_vector/storage --rm ml-sha/hide_dataset_docker

###### Train OD

python3 train.py --exp od --subexp od --device 11

###### Run experiment: scene detection

python3 scene_detection.py --exp scene_detection --subexp three_same --batch_size 1 --device 11
python3 scene_detection.py --exp scene_detection --subexp cross --batch_size 1 --device 11


