#!/bin/bash

MODEL_PATH="/home/models/Qwen2.5-Omni-7B-AWQ"
MODEL_NAME="qwen2.5-omni-7b"
PORT=1314
ARGS="image=1,audio=10"

while [[ "$1" != "" ]]; do
  case $1 in
    -m )
      shift
      MODEL_PATH=$1
      ;;
    -n )
      shift
      MODEL_NAME=$1
      ;;
    -p )
      shift
      PORT=$1
      ;;
    -args )
      shift
      ARGS=$1
      ;;
    --help )
      echo "./model_startup.sh [-m MODEL_PATH] [-n MODEL_NAME] [-p PORT] [-args 'image and audio args list \"image=1,audio=5\"']"
      exit 0
      ;;
    * )
      echo "未知参数 $1"
      exit 1
      ;;
  esac
  shift
done

# 阿里云：
conda activate vllm-env
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --host 0.0.0.0 --port $PORT \
  --dtype float16 \
  --enable-multimodal \
  --task embed \
  --max-model-len 9031 \
  --trust-remote-code \
  --limit-mm-per-prompt $ARGS \
  --gpu-memory-utilization 0.75 \
  --tensor-parallel-size 2 \
  --max-num-seqs 3 \
  --max-num-batched-tokens 9031 \
  --enforce-eager \
  --swap-space 36

# 公司服务器
#conda activate vllm-env
#python3 -m vllm.entrypoints.openai.api_server \
#  --model $MODEL_MODEL_PATH \
#  --served-model-name $MODEL_NAME \
#  --host 0.0.0.0 --port $PORT \
#  --dtype float16 \
#  --quantization awq \
#  --enable-multimodal \
#  --task embed \
#  --max-model-len 9031 \
#  --trust-remote-code \
#  --limit-mm-per-prompt $ARGS \
#  --gpu-memory-utilization 0.75 \
#  --tensor-parallel-size 2 \
#  --max-num-seqs 5 \
#  --max-num-batched-tokens 9031 \
#  --enforce-eager \
#  --swap-space 36
