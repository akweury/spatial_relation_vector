FROM nvcr.io/nvidia/pytorch:22.06-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# install dependencies for opencv
RUN apt-get install ffmpeg libsm6 libxext6  -y

# download the code and install the dependencies
RUN cd /root \
	&& git clone https://github.com/akweury/spatial_relation_vector.git \
	&& cd spatial_relation_vector \
	&& pip uninstall opencv-python \
	&& pip uninstall opencv-contrib-python \
	&& pip uninstall opencv-contrib-python-headless \
	&& pip install opencv-python==4.5.5.64  

RUN cd /root/spatial_relation_vector \ 
	&& git pull
