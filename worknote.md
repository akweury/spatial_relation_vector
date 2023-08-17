
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
python3 scene_detection.py --exp scene_detection --subexp three_same_3 --dataset single_pattern --batch_size 1 --device 6 --top_data 35
python3 scene_detection.py --exp scene_detection --subexp check_mark_4 --dataset single_pattern  --top_data 35 --device 6


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



###### group 7
python3 scene_detection.py --exp scene_detection --subexp letter_L --dataset alphabet --top_data 35 --device 10
python3 scene_detection.py --exp scene_detection --subexp letter_Y --dataset alphabet --top_data 35 --device 10

###### group 9
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_C --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_I --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_J --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_T --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_V --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_X --device 10

###### Group 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_K --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_N --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_O --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_P --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_U --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_Z --device 10

###### Group 11
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_A --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_E --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_F --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_M --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_S --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 26  --subexp letter_W --device 3

###### Group 12
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_B --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_D --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_G --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_H --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_Q --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 35  --subexp letter_R --device 10
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 100  --subexp letter_E --device 4
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 100  --subexp letter_D --device 5
python3 scene_detection.py --exp scene_detection --dataset alphabet --top_data 40  --subexp perpendicular --device 14

###### custom scenes
python3 scene_detection.py --exp scene_detection --subexp diagonal --dataset custom_scenes --top_data 35 --device 6 
python3 scene_detection.py --exp scene_detection --subexp close --dataset custom_scenes --top_data 35 --device 6 
python3 scene_detection.py --exp scene_detection --subexp red_cube_and_random_sphere --dataset custom_scenes --top_data 35 --device 10 
python3 scene_detection.py --exp scene_detection --subexp square --dataset custom_scenes --top_data 8 --device 10 



