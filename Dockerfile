ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && \
    apt-get install -y sudo
RUN apt-get install -y ffmpeg libsm6 libxext6 wget
RUN apt-get install -y vim git

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt



