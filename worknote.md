
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

python3 train.py --exp od --subexp od --device 0


###### check GPU stats
gpustat -cp

###### Run experiment: scene detection

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
python3 scene_detection.py --exp scene_detection --subexp letter_L --dataset alphabet --batch_size 1 --device 6 --e 7
python3 scene_detection.py --exp scene_detection --subexp letter_Y --dataset alphabet --batch_size 1 --device 6 --e 7


python3 scene_detection.py --exp scene_detection --subexp letter_A --batch_size 1 --device 6 --e 10
python3 scene_detection.py --exp scene_detection --subexp letter_I_10 --batch_size 1 --device 6 --e 9
python3 scene_detection.py --exp scene_detection --subexp letter_O_8 --batch_size 1 --device 6 --e 8
python3 scene_detection.py --exp scene_detection --subexp letter_Q_10 --batch_size 1 --device 6 --e 10
python3 scene_detection.py --exp scene_detection --subexp letter_E --batch_size 1 --device 1 --e 11
python3 scene_detection.py --exp scene_detection --subexp letter_H --batch_size 1 --device 6 --e 12
python3 scene_detection.py --exp scene_detection --subexp letter_F --batch_size 1 --device 6 --e 9
python3 scene_detection.py --exp scene_detection --subexp letter_K --batch_size 1 --device 6 --e 9

python3 scene_detection.py --exp scene_detection --subexp letter_M --batch_size 1 --device 6 --e 17
python3 scene_detection.py --exp scene_detection --subexp letter_N --batch_size 1 --device 6 --e 13
python3 scene_detection.py --exp scene_detection --subexp letter_T --batch_size 1 --device 6 --e 7
python3 scene_detection.py --exp scene_detection --subexp letter_V --batch_size 1 --device 6 --e 7
python3 scene_detection.py --exp scene_detection --subexp letter_W --batch_size 1 --device 6 --e 13
python3 scene_detection.py --exp scene_detection --subexp letter_Y --batch_size 1 --device 6 --e 7
python3 scene_detection.py --exp scene_detection --subexp letter_Z --batch_size 1 --device 6 --e 10

python3 scene_detection.py --exp scene_detection --subexp letter_B --batch_size 1 --device 6 --e 11
python3 scene_detection.py --exp scene_detection --subexp letter_C --batch_size 1 --device 6 --e 9
python3 scene_detection.py --exp scene_detection --subexp letter_D --batch_size 1 --device 6 --e 9
python3 scene_detection.py --exp scene_detection --subexp letter_G --batch_size 1 --device 6 --e 14

python3 scene_detection.py --exp scene_detection --subexp letter_J --batch_size 1 --device 6 --e 7
python3 scene_detection.py --exp scene_detection --subexp letter_P --batch_size 1 --device 6 --e 10
python3 scene_detection.py --exp scene_detection --subexp letter_S --batch_size 1 --device 6 --e 13
python3 scene_detection.py --exp scene_detection --subexp letter_U --batch_size 1 --device 6 --e 8

python3 scene_detection.py --exp scene_detection --subexp letter_R --batch_size 1 --device 6 --e 12







