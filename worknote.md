
ssh ml-jsha@130.83.185.153
ssh ml-jsha@130.83.185.155
ssh ml-jsha@130.83.185.145
ssh ml-jsha@130.83.185.147
ssh ml-jsha@130.83.42.209
ssh ml-jsha@130.83.42.211

###### Build docker

docker build -t ml-sha/hide_dataset_docker .

###### Run docker

docker run --gpus all -it -v /home/ml-jsha/storage:/root/storage --rm ml-sha/hide_dataset_docker

###### Train OD

python3 train.py --exp od --subexp od --dataset alphabet --device 3

###### Train OD Letter

python3 train_letter_od.py --exp od_letter --subexp letter_L --device 1

###### check GPU stats
gpustat -cp

###### Run experiment: scene detection
python3 scene_detection.py --exp scene_detection --subexp three_same_3 --dataset single_pattern --batch_size 1 --device 6 -gt 3


python3 scene_detection.py --exp scene_detection --subexp check_mark_4 --dataset single_pattern --batch_size 1 --device 5
python3 scene_detection.py --exp scene_detection --subexp check_mark_5 --dataset single_pattern --batch_size 1 --device 5
python3 scene_detection.py --exp scene_detection --subexp check_mark_6 --dataset single_pattern --batch_size 1 --device 5
python3 scene_detection.py --exp scene_detection --subexp six_same_6 --dataset single_pattern --batch_size 1 --device 5
python3 scene_detection.py --exp scene_detection --subexp ten_same_10 --dataset single_pattern --batch_size 1 --device 5

python3 scene_detection.py --exp scene_detection --subexp three_same --batch_size 1 --device 11
python3 scene_detection.py --exp scene_detection --subexp cross --batch_size 1 --device 11
python3 scene_detection.py --exp scene_detection --subexp cross_same --batch_size 1 --device 11 --e 5
python3 scene_detection.py --exp scene_detection --subexp cross_same_8 --batch_size 1 --device 4 --e 8
python3 scene_detection.py --exp scene_detection --subexp cross_color_8 --batch_size 1 --device 7 --e 8
python3 scene_detection.py --exp scene_detection --subexp cross_position_8 --batch_size 1 --device 7 --e 8
python3 scene_detection.py --exp scene_detection --subexp hexagonal_6 --batch_size 1 --device 7 --e 6
python3 scene_detection.py --exp scene_detection --subexp triangle_6 --batch_size 1 --device 4 --e 6

python3 scene_detection.py --exp scene_detection --subexp pentagon_5_logic --batch_size 1 --device 3 --e 5
python3 scene_detection.py --exp scene_detection --subexp cross_position --batch_size 1 --device 4 --e 5

python3 scene_detection.py --exp scene_detection --subexp letter_a_15 --batch_size 1 --device 8 --e 20
python3 scene_detection.py --exp scene_detection --subexp all_letters --batch_size 1 --device 6 --e 5
python3 scene_detection.py --exp scene_detection --subexp letter_i_12 --batch_size 1 --device 6 --e 6


###### group 7
python3 scene_detection.py --exp scene_detection --subexp letter_L --dataset alphabet --batch_size 1 --device 5
python3 scene_detection.py --exp scene_detection --subexp letter_Y --dataset alphabet --batch_size 1 --device 5

###### Group 11
python3 scene_detection.py --exp scene_detection --subexp letter_A --dataset alphabet --device 6
###### Group 12
python3 scene_detection.py --exp scene_detection --subexp letter_B --dataset alphabet --device 6
python3 scene_detection.py --exp scene_detection --subexp letter_L --dataset alphabet --device 6






