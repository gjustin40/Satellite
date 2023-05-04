# Satellite
Remote Sensing 분야를 위한 Deeplearing Training Repo
- Remote Sensing 분야에 대한 관심 많아지고 있는데, 참고자료가 매우 적음
- `torchgeo`와 같은 라이브러리가 있긴 하지만, wrapping이 너무 되어있어서 연구 목적으로 부적절
- 결론은 그냥 내가 너무 불편해서 만든 레포....사용성이 좋지는 않음

## Installation
- 편리를 위해 서버 전체를 `volumne`했기 때문에 경로는 따로 설정할 필요가 있음
- 위험하긴 하지만 너무 편함
```bash
git clone https://github.com/gjustin40/Satellite.git
cd Satellite
docker build -t satellite --no-cache .
docker run -d --name satellite -it --gpus all --shm-size=100G -v {서버dir}:{container 내에 dir}
docker exec -it satellite /bin/bash
cd {repo clone한 곳으로 이동}
```

## Features
여러 라이브러리를 사용하다가 좋다고 생각되는 부분을 가져와서 적용함
- Epoch-based가 아닌 Iteration-based로 구현 (feat. MMSegmentation)
- Model 단위로 수행되고, Model에는 `forward`, `get_loss`, `get_network`, `get_metric` 등 구현
- `DistributedDataParallel(DDP)`를 사용했고, `torchrun`명령어를 이용해 실행 (`./tools/dist_train.sh`)
- 연구 목적이라 다른 편리한 라이브러리처럼 한 큐에 실행되는 구조는 아님. (Get Started 참고)


## Get Started
`Satellite` 라이브러리를 수행하기 위해 사전에 설정해야하는 부분
- to-do
<details>
<summary>토글 접기/펼치기</summary>
<div markdown="1">

</div>
</details>

## Train
```bash
./tools/dist_train.sh {yaml_config_path} {num_gpu}

# example 1
./tools/dist_train.sh configs/config.yaml 2

# example 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/config.yaml 2

# example 3
PORT=29301 CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/config.yaml 2
```
- `CUDA_VISIBLE_DEVICES`: 여러 GPU 중 원하는 GPU만 선택해서 학습 
- `PORT`: DDP를 사용하면 Port가 필요함. 같은 서버 내에서 여러 train을 실행 할 때 다르게 설정해야함



