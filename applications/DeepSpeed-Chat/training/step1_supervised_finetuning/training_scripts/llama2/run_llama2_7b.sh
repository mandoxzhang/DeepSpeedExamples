#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export DS_ACCELERATOR=musa
export NCCL_PROTOS=2
export MUSA_KERNEL_TIMEOUT=1800

deepspeed --hostfile ./hostfile \
    main.py \
   --data_path pvduy/sharegpt_alpaca_oa_vicuna_format \
   --data_split 2,4,4 \
   --model_name_or_path /home/llama2/llama2_config \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --print_loss \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

# Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets pvduy/sharegpt_alpaca_oa_vicuna_format\
#DS_ACCELER
# DS_ACCELERATOR=musa MUSA_KERNEL_TIMEOUT=1800 NCCL_PROTOS=2 LOCAL_RANK=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=10.11.130.1 MASTER_PORT=29500 python main.py --local_rank=0 --data_path Dahoas/rm-static --data_split 2,4,4 --model_name_or_path /home/cai/LM/llama_config --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --max_seq_len 512 --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 4 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --gradient_checkpointing --zero_stage 3 --deepspeed --offload --output_dir ./output_step1_llama2_7b
# DS_ACCELERATOR=musa MUSA_KERNEL_TIMEOUT=1800 NCCL_PROTOS=2 LOCAL_RANK=1 RANK=1 WORLD_SIZE=2 MASTER_ADDR=10.11.130.1 MASTER_PORT=29500 python main.py --local_rank=1 --data_path Dahoas/rm-static --data_split 2,4,4 --model_name_or_path /home/cai/LM/llama_config --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --max_seq_len 512 --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 4 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --gradient_checkpointing --zero_stage 3 --deepspeed --offload --output_dir ./output_step1_llama2_7b