#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29300}

torchrun --nproc_per_node=1 --master_port=$PORT test_code.py
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py $CONFIG