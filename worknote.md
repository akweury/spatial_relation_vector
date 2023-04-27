
###### Build docker

docker build -t ml-sha/hide_dataset_docker .

###### Run docker

docker run --gpus all -it -v /home/ml-jsha/storage:/root/spatial_relation_vector/storage --rm ml-sha/hide_dataset_docker

###### Train OD

python3 train.py --exp od --subexp od --device 11

###### Run experiment: scene detection

python3 scene_detection.py --exp scene_detection --subexp three_same --batch_size 1 --device 11
python3 scene_detection.py --exp scene_detection --subexp cross --batch_size 1 --device 11
python3 scene_detection.py --exp scene_detection --subexp cross_same --batch_size 1 --device 11 --e 5
python3 scene_detection.py --exp scene_detection --subexp cross_same_8 --batch_size 1 --device 7 --e 8
python3 scene_detection.py --exp scene_detection --subexp cross_color_8 --batch_size 1 --device 7 --e 8
python3 scene_detection.py --exp scene_detection --subexp cross_position_8 --batch_size 1 --device 7 --e 8

python3 scene_detection.py --exp scene_detection --subexp pentagon_5_logic --batch_size 1 --device 4 --e 5
python3 scene_detection.py --exp scene_detection --subexp cross_position --batch_size 1 --device 4 --e 5
python3 scene_detection.py --exp scene_detection --subexp letter_a_15 --batch_size 1 --device 3 --e 20




