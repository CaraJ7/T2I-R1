#!/bin/bash

cd t2i-r1/src
RUN_NAME="t2i-r1"

export DEBUG_MODE="true"
export LOG_PATH="./outputs/debug.txt"
# export NCCL_DEBUG=INFO

QWEN_PATH="deepseek-ai/Janus-Pro-7B"
HF_DATASET="../../../data/geneval_and_t2i_data_final.json" 
OUTPUT_DIR="janus/outputs/${RUN_NAME}" 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nproc_per_node="8" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="12345" \
open_r1/grpo.py --use_vllm False \
--deepspeed "../configs/zero3.json" \
--output_dir $OUTPUT_DIR \
--model_name_or_path $QWEN_PATH \
--dataset_name $HF_DATASET \
--max_prompt_length 512 \
--max_completion_length 1024 \
--temperature 1.0 \
--num_generations 8 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--logging_steps 1 \
--bf16  \
--torch_dtype bfloat16 \
--report_to wandb \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_steps 1600 \
--run_name $RUN_NAME \
--save_steps 400 \
--new_generations_image 1 \
--image_token_num_per_image 576 \
--cfg_weight 5 \
--reasoning_prompt_path ../../../data/prompt/reasoning_prompt.txt \
--reward_funcs hps git gdino orm \
--beta 0.01 \
--tf32 true \
--learning_rate 1e-6 \
--hps_ckpt_path ../../../reward_weight/HPS_v2.1_compressed.pt \
--git_ckpt_path ../../../reward_weight/git-large-vqav2 \
--gdino_ckpt_path ../../../reward_weight/groundingdino_swint_ogc.pth \
--gdino_config_path utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
--orm_ckpt_path ../../../reward_weight/ORM-T2I-R1 \
