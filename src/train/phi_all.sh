#!/bin/bash
export PYTHONNOUSERSITE=1
export TORCH_CPP_LOG_LEVEL=WARNING 
export NCCL_DEBUG=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_VISIBLE_DEVICES="6,7"

export WANDB_API_KEY="WANDB_API_KEY"
export HF_TOKEN="HF_TOKEN"


# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
# --num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
# sft.py \
# training_configs/phi4_sft.yaml \
# 2>&1 | tee logs/phi4_sft.log

# sleep 20

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25679 --config_file training_configs/deepspeed_zero2.yaml \
--num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
sdpo.py \
training_configs/phi4_sdpo.yaml \
2>&1 | tee logs/phi4_sdpo.log

sleep 20

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25680 --config_file training_configs/deepspeed_zero2.yaml \
--num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
dpo.py \
training_configs/phi4_dpo.yaml \
2>&1 | tee logs/phi4_dpo.log

sleep 20

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25680 --config_file training_configs/deepspeed_zero2.yaml \
--num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
sdpo.py \
training_configs/phi4_grpo_sdpo.yaml \
2>&1 | tee logs/phi4_grpo_sdpo.log