ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && \
    apt-get install -y sudo
RUN apt-get install -y ffmpeg libsm6 libxext6 wget
RUN apt-get install -y vim

ARG HOST_UID=1002
ARG HOST_GID=1002
RUN groupadd -g ${HOST_GID} nonrootgroup && \
    useradd -m -s /bin/bash -u ${HOST_UID} -g ${HOST_GID} sakong

RUN usermod -aG sudo sakong && \
    echo "sakong:sakong" | chpasswd
RUN echo "sakong ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sakong

USER sakong
WORKDIR /

ENV PATH="/home/sakong/.local/bin:${PATH}"
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt



