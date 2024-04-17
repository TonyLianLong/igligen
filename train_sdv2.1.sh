#!/bin/bash -x
# Model Name
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

# GPU Settings
NUM_GPUS=4
PARALLEL_PORT=21019
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Determine whether to use multi_gpu based on NUM_GPUS
if [ $NUM_GPUS -eq 1 ]; then
  MULTI_GPU=""
else
  MULTI_GPU="--multi_gpu"
fi

# Training Setting
BATCH_SIZE_SINGLE_GPU=8
NUM_WORKERS=16
DATA_CONFIG_PATH="dataset/sam_full_boxtext2img.yaml"
EXP_NAME=gligen_sdv2.1_bs32_sam

# Run scripts
accelerate launch --multi_gpu --num_processes=$NUM_GPUS --mixed_precision="fp16" \
  --num_machines 1 --dynamo_backend=no --main_process_port $PARALLEL_PORT train_text_to_image_gligen_sam.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config $DATA_CONFIG_PATH \
  --resolution=512 \
  --train_batch_size $BATCH_SIZE_SINGLE_GPU \
  --gradient_accumulation_steps=1 \
  --mixed_precision="fp16" \
  --max_train_steps=500000 \
  --learning_rate=5.e-05 \
  --adam_weight_decay 0.0 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=1000 \
  --output_dir="logs/${EXP_NAME}" \
  --report_to=wandb \
  --dataloader_num_workers $NUM_WORKERS \
  --validation_steps=500 \
  --enable_flash_attention \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 2 \
  --prob_use_caption 0.5 \
  --prob_use_boxes 0.9 \
  --no_caption_only \
  --resume_from_checkpoint latest
