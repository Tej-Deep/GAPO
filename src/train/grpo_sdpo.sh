#!/bin/bash
export PYTHONNOUSERSITE=1
export TORCH_CPP_LOG_LEVEL=WARNING 
export NCCL_DEBUG=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# python -c "import sys; print('\n'.join(sys.path))"

export WANDB_API_KEY="WANDB_API_KEY"
export HF_TOKEN="HF_TOKEN"

# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
# --num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
# sdpo.py \
# training_configs/qwen3_grpo_sdpo.yaml \
# 2>&1 | tee logs/qwen3_grpo_sdpo.log

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
--num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
sdpo.py \
training_configs/llama32_grpo_sdpo.yaml \
2>&1 | tee logs/llama32_grpo_sdpo.log

sleep 20

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
--num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
sdpo.py \
training_configs/llama32_grpo_sdpo_2.yaml \
2>&1 | tee logs/llama32_grpo_sdpo_2.log

# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
# --num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
# sdpo.py \
# training_configs/gemma3_grpo_sdpo.yaml \
# 2>&1 | tee logs/gemma3_grpo_sdpo.log